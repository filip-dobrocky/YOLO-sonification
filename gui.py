import os
import signal

from Qt import QtCore, QtWidgets
from NodeGraphQt import NodeGraph, BaseNode, NodeBaseWidget, NodesPaletteWidget
from text_completer import CompleterTextEdit
import qdarktheme
import cv2

import nodes

tracker = None


def list_camera_ports():
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6:  # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(str(dev_port))
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                working_ports.append(str(dev_port))
            else:
                available_ports.append(str(dev_port))
        dev_port += 1
    return working_ports


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.graph = NodeGraph()
        self.graph.set_context_menu_from_file('./hotkeys.json')

        self.graph.register_nodes([*nodes.synth_nodes,
                                   nodes.ClassesRangeSpecifierNode,
                                   nodes.ClassesTextSpecifierNode,
                                   nodes.ObjectParameterNode,
                                   nodes.TextParameterScalingNode,
                                   nodes.PitchQuantizerNode])

        self.graph.property_changed.connect(nodes.property_changed)
        self.graph.port_connected.connect(nodes.port_connected)
        self.graph.port_disconnected.connect(nodes.port_disconnected)
        self.graph.node_created.connect(nodes.node_created)
        self.graph.session_changed.connect(self.session_changed)

        graph_widget = self.graph.widget

        # TODO: load default

        self.graph.auto_layout_nodes()
        self.graph.clear_selection()
        self.graph.fit_to_selection()

        nodes_palette = NodesPaletteWidget(node_graph=self.graph)
        nodes_palette.set_category_label('nodeGraphQt.nodes', 'Utilities')
        nodes_palette.set_category_label('nodes.synths', 'Synth')
        nodes_palette.set_category_label('nodes.classes', 'Object Classes')
        nodes_palette.set_category_label('nodes.parameters', 'Object Parameters')

        tracker_panel = TrackerPanel()

        # TODO: layout -> QSplitter
        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(graph_widget, 3)
        v_layout.addWidget(nodes_palette)

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addLayout(tracker_panel)
        h_layout.addLayout(v_layout, 7)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(h_layout)

    def session_changed(self, session):
        nodes = self.graph.all_nodes()
        for n in nodes:
            self.graph.node_created.emit(n)
            if not isinstance(n, BaseNode):
                continue
            for input in n.inputs().values():
                for output in input.connected_ports():
                    self.graph.port_connected.emit(input, output)


class TrackerPanel(QtWidgets.QVBoxLayout):
    def __init__(self):
        super(TrackerPanel, self).__init__()

        # Video Source
        self.source_box = QtWidgets.QGroupBox('Video Source')
        self.cam_combo = QtWidgets.QComboBox()
        self.open_button = QtWidgets.QPushButton('Open')
        self.cam_radio = QtWidgets.QRadioButton('Camera adapter')
        self.cam_radio.toggled.connect(self.cam_source_toggled)
        self.file_radio = QtWidgets.QRadioButton('Video file')
        self.file_radio.toggled.connect(self.file_source_toggled)

        self.cam_combo.addItems(list_camera_ports())
        self.cam_combo.currentTextChanged.connect(self.cam_source_changed)
        self.cam_source_changed(self.cam_combo.currentText())
        self.cam_radio.setChecked(True)
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.load_file)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.cam_radio)
        vbox.addWidget(self.cam_combo)
        vbox.addWidget(self.file_radio)
        vbox.addWidget(self.open_button)
        self.source_box.setLayout(vbox)

        # Video Controls
        self.controls_box = QtWidgets.QGroupBox('Video Controls')
        self.video_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal, minimum=0, maximum=100)
        self.video_slider.valueChanged.connect(self.video_slider_moved)
        self.play_button = QtWidgets.QPushButton('Play', checkable=True, checked=False)
        self.play_button.toggled.connect(self.play_toggled)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_slider)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.video_slider, 5)
        hbox.addWidget(self.play_button)
        self.controls_box.setLayout(hbox)
        self.controls_box.setEnabled(False)

        # Tracker Settings
        self.tracker_box = QtWidgets.QGroupBox('Tracker Settings')
        self.classes_checkbox = QtWidgets.QCheckBox('All Classes')
        class_names = tracker.names if tracker is not None else []
        self.classes_text = CompleterTextEdit(keywords=class_names, placeholderText='Class Names', maximumHeight=120)
        self.conf_label = QtWidgets.QLabel('Confidence Threshold')
        self.conf_slider = LabeledFloatSlider(value=tracker.conf_threshold)
        self.iou_label = QtWidgets.QLabel('IOU Threshold')
        self.iou_slider = LabeledFloatSlider(value=tracker.iou_threshold)
        self.size_label = QtWidgets.QLabel('Image Size')
        self.size_slider = LabeledSizeSlider(value=tracker.image_size)
        self.classes_checkbox.toggled.connect(self.classes_toggled)
        self.classes_checkbox.setChecked(True)
        self.iou_slider.valueChanged.connect(self.iou_changed)
        self.conf_slider.valueChanged.connect(self.conf_changed)
        self.size_slider.valueChanged.connect(self.img_size_changed)
        self.classes_text.textChanged.connect(self.classes_changed)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.classes_checkbox)
        vbox.addWidget(self.classes_text)
        vbox.addWidget(self.conf_label)
        vbox.addWidget(self.conf_slider)
        vbox.addWidget(self.iou_label)
        vbox.addWidget(self.iou_slider)
        vbox.addWidget(self.size_label)
        vbox.addWidget(self.size_slider)
        vbox.addStretch(0)
        self.tracker_box.setLayout(vbox)

        self.addWidget(self.source_box)
        self.addWidget(self.controls_box)
        self.addWidget(self.tracker_box)

    def cam_source_toggled(self, checked):
        self.cam_combo.setEnabled(checked)
        if checked:
            self.cam_source_changed(self.cam_combo.currentText())

    def file_source_toggled(self, checked):
        self.open_button.setEnabled(checked)
        self.controls_box.setEnabled(checked)

    def cam_source_changed(self, source):
        if tracker is not None and len(source):
            tracker.load_video(source)

    def load_file(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilter('Videos (*.mp4 *.avi *.mkv)')
        if dialog.exec_():
            files = dialog.selectedFiles()
            tracker.load_video(files[0])
        self.timer.start()
        self.play_button.toggle()

    def play_toggled(self, checked):
        self.play_button.setText('Stop' if checked else 'Play')
        tracker.running = checked

    def video_slider_moved(self, value):
        tracker.set_video_pos(value/100)

    def update_slider(self):
        if tracker is not None:
            self.video_slider.blockSignals(True)
            self.video_slider.setSliderPosition(tracker.video_progress)
            self.video_slider.blockSignals(False)

    def iou_changed(self, value):
        tracker.iou_threshold = value

    def conf_changed(self, value):
        tracker.conf_threshold = value

    def img_size_changed(self, value):
        tracker.image_size = value

    def classes_toggled(self, checked):
        self.classes_text.setEnabled(not checked)
        if checked:
            tracker.classes = None
        else:
            self.classes_changed()

    def classes_changed(self):
        text = self.classes_text.toPlainText()
        names = text.split(',')
        all_names = [x.lower().replace(' ', '') for x in tracker.names]
        classes = []
        for n in names:
            try:
                i = all_names.index(n.lower().replace(' ', ''))
                classes.append(i)
            except ValueError:
                print('Invalid name', n)
        tracker.classes = classes


class LabeledFloatSlider(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(self, value=0.0):
        super(LabeledFloatSlider, self).__init__()
        self.label = QtWidgets.QLabel()
        self.slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal, minimum=0, maximum=100,
                                        value=int(value * 100))
        self.slider.valueChanged.connect(self.value_changed)
        self.value_changed(self.slider.value())
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def value_changed(self, value):
        float_val = value / 100.0
        self.label.setText(f'{float_val:.2f}')
        self.valueChanged.emit(float_val)


class LabeledSizeSlider(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(self, value=640):
        super(LabeledSizeSlider, self).__init__()
        self.label = QtWidgets.QLabel()
        self.slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal, minimum=120, maximum=1080, value=value,
                                        singleStep=20)
        self.slider.valueChanged.connect(self.value_changed)
        self.value_changed(self.slider.value())
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def value_changed(self, value):
        step = self.slider.singleStep()
        if remainder := value % step:
            self.slider.setValue(value - remainder + (step if remainder >= step / 2 else 0))
            return
        self.label.setText(str(value))
        self.valueChanged.emit(value)


def run():
    nodes.tracker = tracker
    app = QtWidgets.QApplication()
    qdarktheme.setup_theme()
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    run()
