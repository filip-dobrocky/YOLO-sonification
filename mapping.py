from tracker import Object
import synths
import supriya
from typing import Callable, List


class SynthMapping:
    def __init__(self, object_classes: List[int], synth_def):
        self.object_classes = object_classes
        self.synth_def = synth_def


class ParameterMapping:
    def __init__(self,
                 synth_def: str,
                 synth_attr: str,
                 obj_attr: str,
                 scaling: Callable[[float], float] = (lambda x: x)):
        self.synth_def = synth_def
        self.synth_attr = synth_attr
        self.obj_attr = obj_attr
        self.scaling = scaling

    def apply(self, synth: supriya.Synth, object: Object):
        if self.synth_def is not None and synth.synthdef.name != self.synth_def:
            return

        synth[self.synth_attr] = self.scaling(getattr(object, self.obj_attr))
