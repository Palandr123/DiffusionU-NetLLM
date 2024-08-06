import numpy as np
from numpy.typing import NDArray
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL.Image import Image
import inflect

from src.detectors.base import Detector
from src.utils import class_aware_non_maximum_suppression, post_process

p = inflect.engine()


class OWLViTv2Detector(Detector):
    def __init__(
        self,
        device: str | torch.device,
    ):
        super().__init__()

        self.processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        owl_vit_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.model = owl_vit_model.eval().to(device)

    def detect(
        self,
        image: Image,
        mode: str,
        device: torch.device | str,
        score_threshold: float,
        nms_threshold: float,
    ) -> list[tuple[str, list[float]]]:
        if mode == "attribute":
            target_objects = [x for x in self.attribute_count]
        elif mode == "primitive":
            target_objects = [x for x in self.primitive_count]
        if len(target_objects) == 0:
            return []

        texts = [[f"image of {p.a(obj)}" for obj in target_objects]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        width, height = image.size
        target_sizes = torch.Tensor([[height, width]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0
        )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)
        boxes = boxes.cpu().detach()
        boxes = np.array(
            [
                [x_min / width, y_min / height, x_max / width, y_max / height]
                for (x_min, y_min, x_max, y_max), score in zip(boxes, scores)
                if score >= score_threshold
            ]
        )
        labels = np.array(
            [
                label.cpu().numpy()
                for label, score in zip(labels, scores)
                if score >= score_threshold
            ]
        )
        scores = np.array(
            [
                score.cpu().detach().numpy()
                for score in scores
                if score >= score_threshold
            ]
        )
        scores, labels, boxes = class_aware_non_maximum_suppression(
            scores, labels, boxes, nms_threshold
        )

        results = []
        for box, label in zip(boxes, labels):
            box = box.tolist()
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            box = post_process(box)
            results.append(
                (
                    f"{target_objects[label]}",
                    box,
                )
            )
        return results

    def __call__(
        self,
        object_lists: list[tuple[str, list[str | None]]],
        image: Image,
        device: torch.device | str,
        attribute_threshold: float = 0.6,
        primitive_threshold: float = 0.2,
        nms_threshold: float = 0.5,
    ) -> dict[str, list[list[float]]]:
        self.register_objects(object_lists)
        attribute_objects = self.detect(
            image, "attribute", device, attribute_threshold, nms_threshold
        )
        primitive_objects = self.detect(
            image, "primitive", device, primitive_threshold, nms_threshold
        )
        results = []
        results.extend(attribute_objects)
        results.extend(primitive_objects)
        results_dict: dict[str, list[list[float]]] = {}
        for (name, box) in results:
            if name in results_dict:
                results_dict[name].append(box)
            else:
                results_dict[name] = [box]
        return results_dict
