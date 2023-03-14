from Qt import QtCore, QtWidgets
from NodeGraphQt import BaseNode, NodeBaseWidget

import math

import synths
import mapping


class SynthNode(BaseNode):
    __identifier__ = 'nodes.synths'
    NODE_NAME = 'Synth'

    def add_inputs(self):
        if self.synthdef is not None:
            self.add_input('classes')
        parameters = set(self.synthdef.parameter_names) \
            if self.synthdef is not None else {'pan', 'level', 'depth'}
        ignored = {'fx_bus', 'out_bus', 'gate', 'buffer'}
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
        # self.label.setStyleSheet('color: lightgrey')

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


class ClassesSpecifierNode(BaseNode):
    __identifier__ = 'nodes.classes'
    NODE_NAME = 'Object Classes'

    def __init__(self):
        super(ClassesSpecifierNode, self).__init__()
        self.add_output('classes')

    @property
    def classes(self):
        raise NotImplementedError


class ClassesRangeSpecifierNode(ClassesSpecifierNode):
    def __init__(self):
        super(ClassesRangeSpecifierNode, self).__init__()

        node_widget = ClassesSliderWrapper(self.view)
        self.add_custom_widget(node_widget, tab='Custom')

    @property
    def classes(self):
        start, stop = self.get_property('classes_range')
        return range(start, stop + 1)


class ObjectParameterNode(BaseNode):
    __identifier__ = 'nodes.parameters'
    NODE_NAME = 'Object Parameter'

    def __init__(self):
        super(ObjectParameterNode, self).__init__()
        self.add_input('scaling')
        self.add_output('parameter')

        self.parameters = ['x', 'y', 'area', 'speed']
        self.add_combo_menu('parameter', 'Parameter', self.parameters)

    @property
    def base_scaling(self):
        norm_dict = {'x': mapping.norm_x,
                     'y': mapping.norm_y,
                     'area': mapping.norm_area,
                     'speed': mapping.norm_speed}
        return norm_dict[self.get_property('parameter')]

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
    NODE_NAME = 'Parameter Scaling'

    def __init__(self):
        super(ParameterScalingNode, self).__init__()
        self.add_input('scaling')
        self.add_output('scaling')
        self.add_text_input('function_str', 'f(x)')
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
    if isinstance(node, ParameterScalingNode):
        if name == 'function_str':
            node.function = parse_function(value)
            for n in node.connected_param_nodes:
                for synth, params in n.connected_parameters.items():
                    for p in params:
                        mapping.modify_parameter_mapping(synth, p, scaling=n.scaling)
    elif isinstance(node, ClassesSpecifierNode):
        if name == 'classes_range':
            port = node.get_output('classes')
            for i in node.connected_output_nodes()[port]:
                if isinstance(i, SynthNode):
                    mapping.remove_synth_mapping(i.synthdef)
                    mapping.add_synth_mapping(node.classes, i.synthdef)
    elif isinstance(node, ObjectParameterNode):
        if name == 'parameter':
            for synth, params in node.connected_parameters.items():
                for p in params:
                    mapping.modify_parameter_mapping(synth, p, obj_attr=value)


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
    input.disconnect_from(output, emit=False)


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

