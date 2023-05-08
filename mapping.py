from tracker import Object
import synths
import supriya
import face_tracking as ft
from typing import Callable, List
import logging


norm_x = lambda x: x
norm_y = lambda x: x
norm_area = lambda x: x
norm_speed = lambda x: max(0, min(x / 50, 1.0))
to_pan = lambda x: max(-1.0, min(2*x-1, 1.0))
cls_to_buf = lambda x: synths.random_class_buffer(x)
sex_to_buf = lambda x: synths.random_sex_buffer(ft.gender_labels[x])
emo_to_buf = lambda x, y: synths.random_emotion_buffer(ft.gender_labels[x], ft.emotion_labels[y])

synth_mappings = list()
param_mappings = list()


class SynthMapping:
    def __init__(self, object_classes: List[int], synthdef: supriya.SynthDef):
        self.object_classes = object_classes
        self.synthdef = synthdef

    def __eq__(self, other):
        return isinstance(other, SynthMapping) and self.synthdef.name == other.synthdef.name


class ParameterMapping:
    def __init__(self,
                 synthdef: supriya.SynthDef,
                 synth_attr: str,
                 obj_attr: str = None,
                 scaling: Callable[[float], float] = (lambda x: x)):
        self.synthdef = synthdef
        self.synth_attr = synth_attr
        self.obj_attr = obj_attr
        self.scaling = scaling

    def __eq__(self, other):
        if isinstance(other, ParameterMapping):
            if self.synthdef is None or other.synthdef is None:
                return self.synthdef is other.synthdef and self.synth_attr == other.synth_attr
            else:
                return self.synthdef.name == other.synthdef.name and self.synth_attr == other.synth_attr
        return False

    def apply(self, synth: supriya.Synth, object: Object):
        if self.synth_attr == 'buffer_id':
            if object.params_changed():
                buf_id, sample_rate = None, None
                if self.obj_attr == 'sex_id':
                    buf_id, sample_rate = sex_to_buf(self.scaling(object.sex_id))
                elif self.obj_attr == 'emo_id':
                    buf_id, sample_rate = emo_to_buf(object.sex_id, self.scaling(object.emo_id))
                if buf_id is not None:
                    synth['buffer_id'] = buf_id
                    if synth.synthdef == synths.player:
                        synth['sample_rate'] = sample_rate
                    return
            elif self.obj_attr in ['sex_id', 'emo_id']:
                return

        synth[self.synth_attr] = self.scaling(getattr(object, self.obj_attr))


def mapping_applies(mapping: ParameterMapping, synth: supriya.Synth):
    # is mapping relevant to this synth?
    if mapping.synthdef is not None:
        return synth.synthdef.name == mapping.synthdef.name
    # False if generic mapping overriden
    return not any(m.synthdef.name == synth.synthdef.name and mapping.synth_attr == m.synth_attr
                   for m in param_mappings if m.synthdef is not None)


def chain(f1: Callable[[float], float], f2: Callable[[float], float]):
    return lambda x: f1(f2(x))


def add_synth_mapping(object_classes: List[int], synthdef: supriya.SynthDef):
    mapping = SynthMapping(object_classes, synthdef)
    if any([m == mapping for m in synth_mappings]):
        logging.info('Attempted duplicate synth mapping.')
        return False
    logging.info(f'Map synth {synthdef.name if synthdef is not None else "any"} to classes {object_classes}.')
    synth_mappings.append(mapping)
    return True


def remove_synth_mapping(synthdef: supriya.SynthDef):
    mapping = None
    for m in synth_mappings:
        if synthdef is None:
            if m.synthdef is None:
                mapping = m
                break
        elif m.synthdef.name == synthdef.name:
            mapping = m
            break
    if mapping is not None:
        logging.info(f'Unmap synth {synthdef.name if synthdef is not None else "any"}.')
        synth_mappings.remove(mapping)


def add_parameter_mapping(synthdef: supriya.SynthDef, synth_attr: str, obj_attr: str,
                          scaling: Callable[[float], float] = (lambda x: x)):
    adjusted_scaling = chain(to_pan, scaling) if synth_attr == 'pan' else scaling
    mapping = ParameterMapping(synthdef, synth_attr, obj_attr, adjusted_scaling)
    if any([mapping == m for m in param_mappings]):
        logging.info('Attempted duplicate parameter mapping.')
        return False
    param_mappings.append(mapping)
    logging.info(f'Map parameter {synth_attr} of synth {synthdef.name if synthdef is not None else "any"}'
                 f' to object parameter {obj_attr} with scaling {scaling}.')
    return True


def remove_parameter_mapping(synthdef: supriya.SynthDef, synth_attr: str):
    mapping = None
    for m in param_mappings:
        if (synthdef is None and m.synthdef is None) or (synthdef is not None and m.synthdef.name == synthdef.name):
            if m.synth_attr == synth_attr:
                mapping = m
                break
    if mapping is not None:
        logging.info(f'Unmap parameter {synth_attr} of synth {synthdef.name if synthdef is not None else "any"}.')
        param_mappings.remove(mapping)


def modify_parameter_mapping(synthdef: supriya.SynthDef, synth_attr: str, obj_attr: str = None,
                             scaling: Callable[[float], float] = None):
    index = None
    for i, m in enumerate(param_mappings):
        if m == ParameterMapping(synthdef, synth_attr):
            index = i
            break
    if index is not None:
        if obj_attr is not None:
            param_mappings[index].obj_attr = obj_attr
            logging.info(f'Modified mapping of {synth_attr}.')
        if scaling is not None:
            adjusted_scaling = chain(to_pan, scaling) if synth_attr == 'pan' else scaling
            param_mappings[index].scaling = adjusted_scaling
            logging.info(f'Modified mapping of {synth_attr}.')
