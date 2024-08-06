class Detector:
    def __init__(self):
        self.object_lists: list[tuple[str, list[str | None]]] = []
        self.primitive_count = {}
        self.attribute_count = {}
        self.pred_primitive_count = {}
        self.pred_attribute_count = {}

    def register_objects(self, object_list: list[tuple[str, list[str | None]]]) -> None:
        """
        Register objects and their attributes from the given object lists.
        """
        # Reset class variables
        self.object_lists = object_list
        self.primitive_count: dict[str, int] = {}
        self.attribute_count: dict[str, int] = {}
        self.pred_primitive_count: dict[str, int] = {}
        self.pred_attribute_count: dict[str, int] = {}

        for name, attribute_list in object_list:
            self.pred_primitive_count[name] = 0
            for attribute in attribute_list:
                if attribute is not None:
                    self.attribute_count[f"{attribute} {name}"] = (
                        self.attribute_count.get(f"{attribute} {name}", 0) + 1
                    )
                    self.pred_attribute_count[f"{attribute} {name}"] = 0
                else:
                    self.primitive_count[name] = self.primitive_count.get(name, 0) + 1
                    self.pred_primitive_count[name] = 0
