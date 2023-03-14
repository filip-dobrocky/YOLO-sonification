import os
import signal

from Qt import QtCore, QtWidgets
from NodeGraphQt import NodeGraph, BaseNode, NodeBaseWidget, NodesPaletteWidget
import qdarktheme

import nodes


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.graph = NodeGraph()
        self.graph.set_context_menu_from_file('./hotkeys.json')

        self.graph.register_nodes([*nodes.synth_nodes,
                              nodes.ClassesRangeSpecifierNode,
                              nodes.ObjectParameterNode,
                              nodes.ParameterScalingNode])

        self.graph.property_changed.connect(nodes.property_changed)
        self.graph.port_connected.connect(nodes.port_connected)
        self.graph.port_disconnected.connect(nodes.port_disconnected)
        self.graph.node_created.connect(nodes.node_created)
        self.graph.session_changed.connect(self.session_changed)

        graph_widget = self.graph.widget

        # load default

        self.graph.auto_layout_nodes()
        self.graph.clear_selection()
        self.graph.fit_to_selection()

        nodes_palette = NodesPaletteWidget(node_graph=self.graph)
        nodes_palette.set_category_label('nodeGraphQt.nodes', 'Utilities')
        nodes_palette.set_category_label('nodes.synths', 'Synth')
        nodes_palette.set_category_label('nodes.classes', 'Object Classes')
        nodes_palette.set_category_label('nodes.parameters', 'Object Parameters')

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(graph_widget, 3)
        layout.addWidget(nodes_palette)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

    def session_changed(self, session):
        nodes = self.graph.all_nodes()
        for n in nodes:
            self.graph.node_created.emit(n)
            if not isinstance(n, BaseNode):
                continue
            for input in n.inputs().values():
                for output in input.connected_ports():
                    self.graph.port_connected.emit(input, output)


if __name__ == '__main__':
    app = QtWidgets.QApplication()
    qdarktheme.setup_theme()
    window = MainWindow()
    window.show()
    app.exec_()
