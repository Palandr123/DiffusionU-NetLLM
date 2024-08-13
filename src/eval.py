from functools import partial
from pathlib import Path
import logging
import random
import time

import torch
import torchvision
import torchvision.transforms.functional as TF
import diffusers
from diffusers.models.attention_processor import Attention
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPImageProcessor, CLIPModel
from PIL import ImageDraw, ImageFont
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import hydra
from omegaconf import DictConfig

from editing_diffusion.diffusion_models import SDXLEditingPipeline
from editing_diffusion.editing import CustomAttentionProcessor
from editing_diffusion.llm_objects import spot_objects
from editing_diffusion.detectors.owlvitv2 import OWLViTv2Detector
from editing_diffusion.editing.edits import position


_SG_RES = 64
SEED = 1
save_aux = True
random.seed(SEED)
torch.manual_seed(SEED)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def resave_aux_key(module, *args, old_key="attn", new_key="last_attn"):
    module._aux[new_key] = module._aux[old_key]


def resize(x):
    return TF.resize(x, _SG_RES, antialias=True)


def stash_to_aux(
    module,
    args,
    kwargs,
    output,
    mode,
    key="last_feats",
    args_idx=None,
    kwargs_key=None,
    fn_to_run=None,
):
    to_save = None
    if mode == "args":
        to_save = input
        if args_idx is not None:
            to_save = args[args_idx]
    elif mode == "kwargs":
        assert kwargs_key is not None
        to_save = kwargs[kwargs_key]
    elif mode == "output":
        to_save = output
    if fn_to_run is not None:
        to_save = fn_to_run(to_save)
    try:
        global save_aux
        if not save_aux:
            len_ = len(module._aux[key])
            del module._aux[key]
            module._aux[key] = [None] * len_ + [to_save]
        else:
            module._aux[key][-1] = module._aux[key][-1].cpu()
            module._aux[key].append(to_save)
    except:
        try:
            del module._aux[key]
        except:
            pass
        module._aux = {key: [to_save]}


def iou_batch(bb_test, bb_gt):

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    start_time = time.time()

    orig_dir = Path(cfg.orig_dir)
    orig_dir.mkdir(parents=True, exist_ok=True)
    edited_dir = Path(cfg.edited_dir)
    edited_dir.mkdir(parents=True, exist_ok=True)

    base = SDXLEditingPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
        torch_dtype=torch.float16,
        variant="fp16",
        use_onnx=False,
    )
    base.to(cfg.device)
    base.scheduler = diffusers.DDPMScheduler.from_config(base.scheduler.config)

    model_name = "google/gemma-7b-it"  # "mistralai/Mistral-7B-Instruct-v0.1" #"google/gemma-7b-it" #"mistralai/Mistral-7B-Instruct-v0.1"
    # Load the model and tokenizer
    llm = AutoModelForCausalLM.from_pretrained(model_name).eval().to(cfg.llm_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    detector = OWLViTv2Detector(cfg.llm_device)

    attn_greenlist = []
    for i in range(0, len(base.unet.up_blocks) - 2):
        for j in range(len(base.unet.up_blocks[i].attentions)):
            base_name = f"up_blocks.{i}.attentions.{j}.transformer_blocks"
            for name, module in (
                base.unet.up_blocks[i].attentions[j].transformer_blocks.named_children()
            ):
                for name_child, _ in module.named_children():
                    if name_child == "attn2":
                        attn_greenlist.append(base_name + f".{name}.{name_child}")
    handle1 = base.unet.up_blocks[2].register_forward_hook(
        partial(stash_to_aux, mode="output"), with_kwargs=True
    )
    handle2 = (
        base.unet.up_blocks[0]
        .attentions[1]
        .transformer_blocks[3]
        .attn2.register_forward_hook(resave_aux_key)
    )
    n_img = 1
    with open(cfg.path, "r") as f:
        lines = f.readlines()
    df = []
    model_ID = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_ID)
    preprocess = CLIPImageProcessor.from_pretrained(model_ID)

    preprocess_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model_dino = AutoModel.from_pretrained('facebook/dinov2-base')
    for i, line in enumerate(lines):
        torch.cuda.empty_cache()
        if i == 1:
            break
        model_params = {
            "max_new_tokens": 200,
        }
        objects = spot_objects(tokenizer, llm, line, cfg.llm_device, **model_params)
        print(objects["objects"])
        handle1.remove()
        handle2.remove()
        global save_aux
        save_aux = True
        for name, block in base.unet.named_modules():
            if isinstance(
                block,
                (
                    diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D,
                    diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D,
                    diffusers.models.unet_2d_blocks.UNetMidBlock2DCrossAttn,
                ),
            ):
                for attn_name, attn in block.named_modules():
                    full_name = name + "." + attn_name
                    if "attn2" not in attn_name or (
                        attn_greenlist and full_name not in attn_greenlist
                    ):
                        continue
                    if isinstance(attn, diffusers.models.attention_processor.Attention):
                        attn.processor = CustomAttentionProcessor(_SG_RES, save_aux)
        handle1 = base.unet.up_blocks[2].register_forward_hook(
            partial(stash_to_aux, mode="output"), with_kwargs=True
        )
        handle2 = (
            base.unet.up_blocks[0]
            .attentions[1]
            .transformer_blocks[3]
            .attn2.register_forward_hook(resave_aux_key)
        )

        logger.info(f"Generating image #{i}")
        generator = torch.Generator(device=cfg.device).manual_seed(SEED)
        out = base(
            prompt=[line] * n_img,
            num_inference_steps=300,
            generator=generator,
            save_aux=save_aux,
            latents=None,
        )
        

        aux = base.get_sg_aux()
        aux_idx = 0
        processed_aux = {
            k: torch.utils._pytree.tree_map(
                lambda x: x[aux_idx : aux_idx + 1].repeat_interleave(n_img, 0).cpu(), v
            )
            for k, v in aux.items()
        }

        logger.info(f"Detecting objects on the image #{i}")
        results_detection = detector(
            objects["objects"], out.images[0], cfg.llm_device, 0.5, 0.3, 0.4
        )

        width, height = 64, 64
        results_new_detection = {}
        for name, bboxes in results_detection.items():
            bboxes_new = []
            for bbox in bboxes:
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h
                x1 *= width
                x2 *= width
                y1 *= height
                y2 *= height
                bboxes_new.append([int(x1), int(y1), int(x2), int(y2)])
            results_new_detection[name] = bboxes_new
        
        img = out.images[0].copy()
        draw = ImageDraw.Draw(img)
        object_count = {}
        a, b = out.images[0].size
        for name, box_list in results_detection.items():
            for box in box_list:
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h
                x1 *= a
                x2 *= a
                y1 *= b
                y2 *= b
                draw.rectangle(((int(x1), int(y1)), (int(x2), int(y2))), outline=(255, 0, 0))
                font = ImageFont.load_default(size=20)
            
                text_size = draw.textbbox((0, 0), f"{name}, id={object_count.get(name, 0)}", font=font)
            
                text_position = (x1, max(y1 - text_size[3] - 2, 0))
                draw.text(text_position, f"{name}, id={object_count.get(name, 0)}", fill='white', font=font)
                object_count[name] = object_count.get(name, 0) + 1
        img.save(str(orig_dir / f"img{i}.PNG"), "PNG")
        if not results_new_detection:
            continue
        manipulated_type = random.choice(list(results_new_detection.keys()))
        manipulated_id = random.randint(
            0, len(results_new_detection[manipulated_type]) - 1
        )
        x1, y1, x2, y2 = results_new_detection[manipulated_type][manipulated_id]
        x_manipulation = random.randint(-x1, width - x2)
        y_manipulation = random.randint(-y1, height - y2)
        print(manipulated_type, manipulated_id, x_manipulation, y_manipulation)

        handle1.remove()
        handle2.remove()
        save_aux = False
        for name, block in base.unet.named_modules():
            if isinstance(
                block,
                (
                    diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D,
                    diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D,
                    diffusers.models.unet_2d_blocks.UNetMidBlock2DCrossAttn,
                ),
            ):
                for attn_name, attn in block.named_modules():
                    full_name = name + "." + attn_name
                    if "attn2" not in attn_name or (
                        attn_greenlist and full_name not in attn_greenlist
                    ):
                        continue
                    if isinstance(attn, Attention):
                        attn.processor = CustomAttentionProcessor(_SG_RES, save_aux)
        handle1 = base.unet.up_blocks[2].register_forward_hook(
            partial(stash_to_aux, mode="output"), with_kwargs=True
        )
        handle2 = (
            base.unet.up_blocks[0]
            .attentions[1]
            .transformer_blocks[3]
            .attn2.register_forward_hook(resave_aux_key)
        )

        torch.cuda.empty_cache()
        sg_edits = {
            "fn": position,
            "words": manipulated_type,
            "mode": "attn",
            "mode_preserve": ["attn", "last_feats"],
            "weight": 20.0,
            "kwargs": {
                "shift": (x_manipulation, y_manipulation),
                "box_orig": results_new_detection[manipulated_type][manipulated_id],
            },
            "w1": 50.0,
            "w2": 20.0,
            "weight_preserve": 20.0,
            "tgt": processed_aux,
        }

        num_inference_steps = 300
        sg_loss_rescale = 1000.0
        sg_grad_wt = 300.0
        sg_t_start = 20
        sg_t_end = 280
        generator = torch.Generator(device=cfg.device).manual_seed(SEED)
        try:
            out_edited = base(
                prompt=[line] * n_img,
                sg_grad_wt=sg_grad_wt,
                edit=sg_edits,
                num_inference_steps=num_inference_steps,
                sg_loss_rescale=sg_loss_rescale,
                debug=False,
                sg_t_start=sg_t_start,
                sg_t_end=sg_t_end,
                generator=generator,
                save_aux=save_aux,
                latents=None,
                detections=results_new_detection,
                target_object=(manipulated_type, manipulated_id),
            )
        except:
            torch.cuda.empty_cache()
            continue
        if out_edited is None:
            continue
        out_edited.images[0].save(str(edited_dir / f"img{i}.PNG"), "PNG")
        results_detection_edited = detector(
            objects["objects"], out_edited.images[0], cfg.llm_device, 0.5, 0.3, 0.4
        )
        matches = {}
        iou_threshold = 0.4
        unmatched_orig = 0
        unmatched_edit = 0
        for key, bboxes in results_detection_edited.items():
            if key not in results_detection:
                unmatched_edit += len(results_detection_edited[key])
                matches[key] = [-1 for _ in bboxes]
                continue
            bboxes = [
                [x, y, x + w, y + h] for x, y, w, h in results_detection_edited[key]
            ]
            bboxes_orig = [
                [x, y, x + w, y + h]
                for i, (x, y, w, h) in enumerate(results_detection[key])
            ]
            if key == manipulated_type:
                x1, y1, x2, y2 = bboxes_orig[manipulated_id]
                bboxes_orig[manipulated_id] = [
                    x1 + x_manipulation / width,
                    y1 + y_manipulation / height,
                    x2 + x_manipulation / width,
                    y2 + y_manipulation / height,
                ]
            iou_matrix = iou_batch(np.array(bboxes), np.array(bboxes_orig))
            if min(iou_matrix.shape) > 0:
                a = (iou_matrix > iou_threshold).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    matched_indices = np.stack(np.where(a), axis=1)
                else:
                    matched_indices = linear_assignment(-iou_matrix)
            else:
                matched_indices = np.empty(shape=(0, 2))

            unmatched_detections = []
            for d, det in enumerate(results_detection_edited[key]):
                if d not in matched_indices[:, 0]:
                    unmatched_detections.append(d)

            unmatched_trackers = []
            for t, trk in enumerate(results_detection[key]):
                if t not in matched_indices[:, 1]:
                    unmatched_trackers.append(t)

            # filter out matched with low IOU
            res = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < iou_threshold:
                    unmatched_detections.append(m[0])
                    unmatched_trackers.append(m[1])
                else:
                    res.append(m.reshape(1, 2))

            if len(res) == 0:
                res = np.empty((0, 2), dtype=int)
            else:
                res = np.concatenate(res, axis=0)
            matches[key] = [-1 for _ in bboxes]
            for x, y in res:
                matches[key][x] = y
            unmatched_orig += len(unmatched_trackers)
            unmatched_edit += len(unmatched_detections)
        similarity_scores = []
        for key in matches.keys():
            for i, match in enumerate(matches[key]):
                if match == -1:
                    continue
                x, y, w, h = [int(i * 1024) for i in results_detection[key][match]]
                cropped_orig = torchvision.transforms.functional.to_tensor(out.images[0])[:, y:y+h, x:x+w]
                image_a = preprocess(torchvision.transforms.functional.to_pil_image(cropped_orig), return_tensors="pt")['pixel_values'][0]

                x, y, w, h = [int(i * 1024) for i in results_detection_edited[key][i]]
                cropped_edited = torchvision.transforms.functional.to_tensor(out_edited.images[0])[:, y:y+h, x:x+w]
                image_b = preprocess(torchvision.transforms.functional.to_pil_image(cropped_edited), return_tensors="pt")['pixel_values'][0]

                # Calculate the embeddings for the images using the CLIP model
                with torch.no_grad():
                    embedding_a = clip_model.get_image_features(image_a.unsqueeze(0))
                    embedding_b = clip_model.get_image_features(image_b.unsqueeze(0))

                # Calculate the cosine similarity between the embeddings
                similarity_scores.append(torch.nn.functional.cosine_similarity(embedding_a, embedding_b))
        cropped_orig = torchvision.transforms.functional.to_tensor(out.images[0])
        image_a = preprocess(torchvision.transforms.functional.to_pil_image(cropped_orig), return_tensors="pt")['pixel_values'][0]

        cropped_edited = torchvision.transforms.functional.to_tensor(out_edited.images[0])
        image_b = preprocess(torchvision.transforms.functional.to_pil_image(cropped_edited), return_tensors="pt")['pixel_values'][0]

        # Calculate the embeddings for the images using the CLIP model
        with torch.no_grad():
            embedding_a = clip_model.get_image_features(image_a.unsqueeze(0))
            embedding_b = clip_model.get_image_features(image_b.unsqueeze(0))
        clip_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)

        dino_scores = []
        for key in matches.keys():
            for i, match in enumerate(matches[key]):
                if match == -1:
                    continue
                x, y, w, h = [int(i * 1024) for i in results_detection[key][match]]
                cropped_orig = torchvision.transforms.functional.to_tensor(out.images[0])[:, y:y+h, x:x+w]
                image_a = preprocess_dino(torchvision.transforms.functional.to_pil_image(cropped_orig), return_tensors="pt")['pixel_values'][0]

                x, y, w, h = [int(i * 1024) for i in results_detection_edited[key][i]]
                cropped_edited = torchvision.transforms.functional.to_tensor(out_edited.images[0])[:, y:y+h, x:x+w]
                image_b = preprocess_dino(torchvision.transforms.functional.to_pil_image(cropped_edited), return_tensors="pt")['pixel_values'][0]

                # Calculate the embeddings for the images using the CLIP model
                with torch.no_grad():
                    embedding_a = model_dino(image_a.unsqueeze(0)).last_hidden_state.mean(dim=1)
                    embedding_b = model_dino(image_b.unsqueeze(0)).last_hidden_state.mean(dim=1)

                # Calculate the cosine similarity between the embeddings
                dino_scores.append(torch.nn.functional.cosine_similarity(embedding_a, embedding_b))
        
        cropped_orig = torchvision.transforms.functional.to_tensor(out.images[0])
        image_a = preprocess_dino(torchvision.transforms.functional.to_pil_image(cropped_orig), return_tensors="pt")['pixel_values'][0]

        cropped_edited = torchvision.transforms.functional.to_tensor(out_edited.images[0])
        image_b = preprocess_dino(torchvision.transforms.functional.to_pil_image(cropped_edited), return_tensors="pt")['pixel_values'][0]

        # Calculate the embeddings for the images using the CLIP model
        with torch.no_grad():
            embedding_a = model_dino(image_a.unsqueeze(0)).last_hidden_state.mean(dim=1)
            embedding_b = model_dino(image_b.unsqueeze(0)).last_hidden_state.mean(dim=1)
        dino_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
        total_object = sum([len(x) for x in matches.values()])
        if manipulated_type not in matches:
            successfull = False
        else:
            successfull = manipulated_id in matches[manipulated_type]
        df.append(
            {
                "manipulated_type": manipulated_type,
                "manipulated_id": manipulated_id,
                "x_manipulation": x_manipulation,
                "y_manipulation": y_manipulation,
                "unmatched_orig": unmatched_orig / (total_object+unmatched_orig) if total_object+unmatched_orig !=0 else 0,
                "unmatched_edit": unmatched_edit / (total_object+unmatched_edit)if total_object+unmatched_edit !=0 else 0,
                "successfull": successfull,
                "CLIP_score_object": np.mean(similarity_scores),
                "CLIP_score_image": clip_score.item(),
                "DINO_score_object": np.mean(dino_scores),
                "DINO_score_image": dino_score.item(),
            }
        )
    pd.DataFrame(df).to_csv("manipulations.csv", index=False)

    logger.info(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
