from Qt import QtCore, QtWidgets
from NodeGraphQt import BaseNode, NodeBaseWidget
from text_completer import CompleterTextEdit

import math

import synths
import mapping
import pitch_quantization

tracker = None


class SynthNode(BaseNode):
    __identifier__ = 'nodes.synths'
    NODE_NAME = 'Synth'

    def add_inputs(self):
        if self.synthdef is not None:
            self.add_input('classes')
        parameters = set(self.synthdef.parameter_names) \
            if self.synthdef is not None else {'pan', 'level', 'depth'}
        ignored = {'fx_bus', 'out_bus', 'gate'}
        for p in parameters - ignored:
            self.add_input(p)


def synth_constructor(self):
    super(SynthNode, self).__init__()
    self.add_inputs()


def create_synth_nodes():
    synthdefs = {'any': None}
    synthdefs.update(dict(zip([x.name for x in synths.synthdefs], synths.synthdefs)))
    nodes = []
    for k, v in synthdefs.items():
        name = k.capitalize()
        node = type(name+'SynthNode', (SynthNode,), {
            '__init__': synth_constructor,
            'synthdef': v,
            'NODE_NAME': name
        })
        nodes.append(node)
    return nodes


synth_nodes = create_synth_nodes()


class ClassesSlider(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ClassesSlider, self).__init__(parent)

        min, max = 0, 79
        slider_style = "QSlider::add-page:horizontal {background:rgba(0,0,0,0);}" \
                       "QSlider::sub-page:horizontal {background:rgba(0,0,0,0);}"
        self.startSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.startSlider.setMinimum(min)
        self.startSlider.setMaximum(max)
        self.startSlider.setValue(min)
        self.startSlider.setStyleSheet(slider_style)
        self.endSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.endSlider.setMinimum(min)
        self.endSlider.setMaximum(max)
        self.endSlider.setValue(max)
        self.endSlider.setStyleSheet(slider_style)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.startSlider)
        layout.addWidget(self.endSlider)


class ClassesSliderWrapper(NodeBaseWidget):
    def __init__(self, parent=None):
        super(ClassesSliderWrapper, self).__init__(parent)
        self.set_name('classes_range')
        self.set_label('Classes range')

        widget = ClassesSlider()
        widget.startSlider.valueChanged.connect(self.on_value_changed)
        widget.startSlider.valueChanged.connect(self.start_changed)
        widget.endSlider.valueChanged.connect(self.end_changed)
        widget.endSlider.valueChanged.connect(self.on_value_changed)
        self.set_custom_widget(widget)

    def start_changed(self):
        widget = self.get_custom_widget()
        if widget.startSlider.value() > widget.endSlider.value():
            widget.endSlider.setValue(widget.startSlider.value())

    def end_changed(self):
        widget = self.get_custom_widget()
        if widget.startSlider.value() > widget.endSlider.value():
            widget.startSlider.setValue(widget.endSlider.value())

    def set_value(self, value):
        widget = self.get_custom_widget()
        widget.startSlider.setValue(value[0])
        widget.endSlider.setValue(value[1])

    def get_value(self):
        widget = self.get_custom_widget()
        widget.label.setText(f'({widget.startSlider.value()}, {widget.endSlider.value()})')
        return widget.startSlider.value(), widget.endSlider.value()


class ClassesTextWrapper(NodeBaseWidget):
    def __init__(self, parent=None):
        super(ClassesTextWrapper, self).__init__(parent)
        self.set_name('classes_text')
        class_names = tracker.names if tracker is not None else []
        widget = CompleterTextEdit(keywords=class_names, placeholderText='Class Names', maximumHeight=120)
        widget.textChanged.connect(self.on_value_changed)
        self.set_custom_widget(widget)

    def set_value(self, text):
        self.get_custom_widget().setPlainText(text)

    def get_value(self):
        return self.get_custom_widget().toPlainText()


class ClassesSpecifierNode(BaseNode):
    __identifier__ = 'nodes.classes'

    def __init__(self):
        super(ClassesSpecifierNode, self).__init__()
        self.add_output('classes')

    @property
    def classes(self):
        raise NotImplementedError


class ClassesRangeSpecifierNode(ClassesSpecifierNode):
    NODE_NAME = 'Classes Range'

    def __init__(self):
        super(ClassesRangeSpecifierNode, self).__init__()

        node_widget = ClassesSliderWrapper(self.view)
        self.add_custom_widget(node_widget, tab='Custom')

    @property
    def classes(self):
        start, stop = self.get_property('classes_range')
        return range(start, stop + 1)


class ClassesTextSpecifierNode(ClassesSpecifierNode):
    NODE_NAME = 'Classes Text'

    def __init__(self):
        super(ClassesTextSpecifierNode, self).__init__()
        node_widget = ClassesTextWrapper(self.view)
        self.add_custom_widget(node_widget, tab='Custom')

    @property
    def classes(self):
        text = self.get_property('classes_text')
        names = text.split(',')
        all_names = [x.lower().replace(' ', '') for x in tracker.names]
        classes = []
        for n in names:
            try:
                i = all_names.index(n.lower().replace(' ', ''))
                classes.append(i)
            except ValueError:
                print('Invalid name', n)
        return classes


class ObjectParameterNode(BaseNode):
    __identifier__ = 'nodes.parameters'
    NODE_NAME = 'Object Parameter'

    def __init__(self):
        super(ObjectParameterNode, self).__init__()
        self.add_input('scaling')
        self.add_output('parameter')

        self.parameters = ['x', 'y', 'area', 'speed', 'class_id', 'emo_id', 'sex_id']
        self.add_combo_menu('parameter', 'Parameter', self.parameters)

    @property
    def base_scaling(self):
        norm_dict = {'x': mapping.norm_x,
                     'y': mapping.norm_y,
                     'area': mapping.norm_area,
                     'speed': mapping.norm_speed}
        param = self.get_property('parameter')
        return norm_dict[param] if param in norm_dict else lambda x: x

    @property
    def scaling(self):
        connected_f = None
        port = self.get_input('scaling')
        for i in self.connected_input_nodes()[port]:
            if isinstance(i, ParameterScalingNode):
                connected_f = i
                break
        if connected_f is not None:
            return mapping.chain(connected_f.chained_function, self.base_scaling)
        return self.base_scaling

    @property
    def connected_parameters(self):
        param_dict = dict()
        port = self.get_output(0)
        for i in set(self.connected_output_nodes()[port]):
            if isinstance(i, SynthNode):
                synth = i.synthdef
                param_dict[synth] = [k.name() for k, v in i.connected_input_nodes().items()
                                     if len(v) and v[0].id == self.id]
        return param_dict


class ParameterScalingNode(BaseNode):
    __identifier__ = 'nodes.parameters'

    def __init__(self):
        super(ParameterScalingNode, self).__init__()
        self.add_input('scaling')
        self.add_output('scaling')
        self.function = lambda x: x

    @property
    def chained_function(self):
        connected_f = None
        port = self.get_input(0)
        for i in self.connected_input_nodes()[port]:
            if isinstance(i, ParameterScalingNode):
                connected_f = i
                break

        if connected_f is not None:
            return mapping.chain(connected_f.chained_function, self.function)
        return self.function

    @property
    def connected_param_nodes(self):
        parameters = list()
        port = self.get_output(0)
        for i in set(self.connected_output_nodes()[port]):
            if isinstance(i, ObjectParameterNode):
                parameters.append(i)
            elif isinstance(i, ParameterScalingNode):
                connected = i.connected_param_nodes
                if len(connected):
                    parameters.extend(connected)
        return parameters


class TextParameterScalingNode(ParameterScalingNode):
    NODE_NAME = 'Parameter Scaling'

    def __init__(self):
        super(TextParameterScalingNode, self).__init__()
        self.add_text_input('function_str', 'f(x)')


class PitchQuantizerNode(ParameterScalingNode):
    NODE_NAME = 'Pitch Quantizer'

    def __init__(self):
        super(PitchQuantizerNode, self).__init__()
        self.add_text_input('ref_f', 'A4', '440')
        self.add_combo_menu('root', 'Root', pitch_quantization.note_names)
        self.add_combo_menu('scale', 'Scale', pitch_quantization.scale_names)
        self.quantizer = None
        self.update_quantizer()

    def update_quantizer(self):
        root = self.get_property('root')
        scale = self.get_property('scale')
        try:
            ref_f = float(str(self.get_property('ref_f')))
        except ValueError:
            ref_f = 440
        self.quantizer = pitch_quantization.Quantizer(root, scale, ref_f)
        self.function = self.quantizer.snap


def parse_function(string):
    try:
        if 'x' not in string:
            raise SyntaxError
        function = eval('lambda x:' + string, math.__dict__)
        if function(1) is None:
            raise SyntaxError
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        print('Incorrect scaling function, setting f(x)=x')
        function = lambda x: x
    return function


def property_changed(node, name, value):
    if isinstance(node, TextParameterScalingNode):
        if name == 'function_str':
            node.function = parse_function(value)
    elif isinstance(node, PitchQuantizerNode):
        node.update_quantizer()
    elif isinstance(node, ClassesSpecifierNode):
        if name in ['classes_range', 'classes_text']:
            port = node.get_output('classes')
            for i in node.connected_output_nodes()[port]:
                if isinstance(i, SynthNode):
                    mapping.remove_synth_mapping(i.synthdef)
                    if len(classes := node.classes):
                        mapping.add_synth_mapping(classes, i.synthdef)
    elif isinstance(node, ObjectParameterNode):
        if name == 'parameter':
            for synth, params in node.connected_parameters.items():
                for p in params:
                    mapping.modify_parameter_mapping(synth, p, obj_attr=value)
    if isinstance(node, ParameterScalingNode):
        for n in node.connected_param_nodes:
            for synth, params in n.connected_parameters.items():
                for p in params:
                    mapping.modify_parameter_mapping(synth, p, scaling=n.scaling)


def port_connected(input, output):
    if isinstance(input.node(), SynthNode):
        if input.name() == 'classes':
            if isinstance(output.node(), ClassesSpecifierNode):
                synthdef = input.node().synthdef
                classes = output.node().classes
                if mapping.add_synth_mapping(classes, synthdef):
                    return
        elif isinstance(output.node(), ObjectParameterNode):
            synthdef = input.node().synthdef
            synth_parameter = input.name()
            object_parameter = output.node().get_property('parameter')
            scaling = output.node().scaling
            if mapping.add_parameter_mapping(synthdef, synth_parameter, object_parameter, scaling):
                return
    elif isinstance(input.node(), ObjectParameterNode):
        if isinstance(output.node(), ParameterScalingNode):
            for synth, params in input.node().connected_parameters.items():
                for p in params:
                    mapping.modify_parameter_mapping(synth, p, scaling=input.node().scaling)
            return
    elif isinstance(input.node(), ParameterScalingNode):
        if isinstance(output.node(), ParameterScalingNode):
            for n in input.node().connected_param_nodes:
                for synth, params in n.connected_parameters.items():
                    for p in params:
                        mapping.modify_parameter_mapping(synth, p, scaling=n.scaling)
            return

    input.node().graph.blockSignals(True)
    input.disconnect_from(output)
    input.node().graph.blockSignals(False)


def port_disconnected(input, output):
    if isinstance(input.node(), SynthNode):
        if input.name() == 'classes' and isinstance(output.node(), ClassesSpecifierNode):
            synthdef = input.node().synthdef
            mapping.remove_synth_mapping(synthdef)
        elif isinstance(output.node(), ObjectParameterNode):
            synthdef = input.node().synthdef
            synth_parameter = input.name()
            mapping.remove_parameter_mapping(synthdef, synth_parameter)
    elif isinstance(input.node(), ObjectParameterNode):
        if isinstance(output.node(), ParameterScalingNode):
            for synth, params in input.node().connected_parameters.items():
                for p in params:
                    mapping.modify_parameter_mapping(synth, p, scaling=input.node().scaling)
    elif isinstance(input.node(), ParameterScalingNode):
        if isinstance(output.node(), ParameterScalingNode):
            for n in input.node().connected_param_nodes:
                for synth, params in n.connected_parameters.items():
                    for p in params:
                        mapping.modify_parameter_mapping(synth, p, scaling=n.scaling)


def node_created(node):
    if isinstance(node, SynthNode):
        graph = node.graph
        other_nodes = set(graph.all_nodes()) - {node}
        if any([type(node) == type(n) for n in other_nodes]):
            graph.delete_node(node)
    if isinstance(node, ParameterScalingNode):
        node.function = parse_function(node.get_property('function_str'))
        for n in node.connected_param_nodes:
            for synth, params in n.connected_parameters.items():
                for p in params:
                    mapping.modify_parameter_mapping(synth, p, scaling=n.scaling)

