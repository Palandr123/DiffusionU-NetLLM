from collections import defaultdict
from typing import Callable, Optional, Any


import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)

from src.editing import search_sequence_numpy
from src.editing.edits import preserve


class SDXLEditingPipeline(StableDiffusionXLPipeline):
    def get_sg_aux(
        self, apply_tree_map: bool = True, transpose: bool = True
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Get auxiliary data for the SelfGuidanceSDXLPipeline.
        This method iterates through named modules in the UNet and collects auxiliary data from them.
        The `apply_tree_map` and `transpose` parameters control the operations applied to the data.
        If `transpose` is True, it transposes the data. If `apply_tree_map` is True, it applies tree_map to the data.

        Args:
            apply_tree_map: bool - whether to apply tree_map. Default is True.
            transpose: bool - whether to transpose the data. Default is True.

        Returns:
            dict[str, dict[str, torch.Tensor]] -tThe auxiliary data.
        """
        auxiliary_data = defaultdict(dict)
        for name, aux_module in self.unet.named_modules():
            try:
                module_auxiliary = aux_module._aux
                if transpose:
                    for key, value in module_auxiliary.items():
                        if apply_tree_map:
                            value = torch.utils._pytree.tree_map(
                                lambda vv: vv.chunk(2)[1] if vv is not None else None,
                                value,
                            )
                        auxiliary_data[key][name] = value
                else:
                    auxiliary_data[name] = module_auxiliary
                    if apply_tree_map:
                        auxiliary_data[name] = {
                            k: torch.utils._pytree.tree_map(
                                lambda vv: vv.chunk(2)[1] if vv is not None else None, v
                            )
                            for k, v in auxiliary_data[name].items()
                        }
            except AttributeError:
                pass
        return auxiliary_data

    def wipe_sg_aux(self):
        for name, aux_module in self.unet.named_modules():
            try:
                del aux_module._aux
            except AttributeError:
                pass

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[str | list[str]] = None,
        negative_prompt_2: Optional[str | list[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator | list[torch.Generator]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        original_size: Optional[tuple[int, int]] = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        target_size: Optional[tuple[int, int]] = None,
        sg_grad_wt=1.0,
        edit: dict | None = None,
        sg_loss_rescale=1000.0,  # prevent fp16 underflow
        debug=False,
        sg_t_start=0,
        sg_t_end=-1,
        save_aux=False,
        detections: dict = {},
        target_object: tuple = (None, None),
    ):
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        do_self_guidance = sg_grad_wt > 0 and edit is not None

        if do_self_guidance:
            prompt_text_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"][
                0
            ]

            if "words" not in edit:
                edit["idxs"] = np.arange(len(prompt_text_ids))
            else:
                words = edit["words"]
                if not isinstance(words, list):
                    words = [words]
                idxs = []
                for word in words:
                    word_ids = self.tokenizer(word, return_tensors="np")["input_ids"]
                    word_ids = word_ids[word_ids < 49406]
                    idxs.append(search_sequence_numpy(prompt_text_ids, word_ids))
                edit["idxs"] = np.concatenate(idxs)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # 7.1 Apply denoising_end
        if (
            denoising_end is not None
            and type(denoising_end) == float
            and denoising_end > 0
            and denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        self.wipe_sg_aux()
        torch.cuda.empty_cache()
        if sg_t_end < 0:
            sg_t_end = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # torch.cuda.empty_cache()
                # expand the latents if we are doing classifier free guidance
                with torch.set_grad_enabled(
                    do_self_guidance
                ):  # , torch.autograd.detect_anomaly():
                    latents.requires_grad_(do_self_guidance)
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )

                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # predict the noise residual
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    ### SELF GUIDANCE
                    if do_self_guidance and sg_t_start <= i < sg_t_end:
                        torch.cuda.empty_cache()
                        sg_aux = self.get_sg_aux(do_classifier_free_guidance)
                        sg_loss = 0
                        if isinstance(edit["mode"], str):
                            key_aux = sg_aux[edit["mode"]]
                        else:
                            key_aux = {"": {k: sg_aux[k] for k in edit["mode"]}}
                        losses = []
    
                        wt = edit.get("weight", 1.0)
                        if wt:
                            tgt = edit.get("tgt", None)
                            if tgt is not None:
                                if isinstance(edit["mode"], str):
                                    tgt = tgt[edit["mode"]]
                                else:
                                    tgt = {"": {k: tgt[k] for k in edit["mode"]}}
                            losses = []
                            for k, v in key_aux.items():
                                loss = edit["fn"](
                                    v,
                                    i=i,
                                    idxs=edit["idxs"],
                                    **edit.get("kwargs", {}),
                                    tgt=tgt[k] if tgt is not None else None,
                                )
                                losses.append(loss)
                            edit_loss = torch.stack(losses).mean()
                            sg_loss += wt * edit_loss

                        losses = []
                        key_aux = sg_aux["last_feats"]
                        tgt = edit.get("tgt", None)
                        tgt = tgt["last_feats"]

                        for name, object_list in detections.items():
                            for id, box in enumerate(object_list):
                                for k, v in key_aux.items():
                                    box_preserve = [2 * x for x in box]
                                    if (
                                        name == target_object[0]
                                        and id == target_object[1]
                                    ):
                                        continue
                                    loss = edit.get(
                                        "weight_preserve", 1.0
                                    ) * preserve(
                                        v,
                                        i=i,
                                        idxs=None,
                                        box_orig=box_preserve,
                                        target_aux=(
                                            tgt[k] if tgt is not None else None
                                        ),
                                    )
                                    losses.append(loss.cpu())

                        if len(losses) != 0:
                            preserve_loss = torch.stack(losses).mean()
                            sg_loss += preserve_loss
                        if not torch.isnan(sg_loss).any():

                            sg_grad = (
                                torch.autograd.grad(sg_loss_rescale * sg_loss, latents)[
                                    0
                                ]
                                / sg_loss_rescale
                            )
                            if not torch.isnan(sg_grad).any():
                                noise_pred = noise_pred + sg_grad_wt * sg_grad
                        if noise_pred.isnan().any():
                            return None
                    latents.detach()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        torch.cuda.empty_cache()
        if not save_aux:
            self.wipe_sg_aux()
        latents = latents.detach()
        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype
            )

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
