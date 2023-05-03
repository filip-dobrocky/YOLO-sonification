

from PySide2 import QtGui
from PySide2.QtCore import Qt
from PySide2.QtGui import QTextCursor
from PySide2.QtWidgets import QCompleter, QPlainTextEdit


class CompleterTextEdit(QPlainTextEdit):
    """Texteditor with autocompletion."""

    def __init__(self, keywords, *args, **kwargs):
        super(CompleterTextEdit, self).__init__(*args, **kwargs)
        completer = QCompleter(keywords)
        completer.activated.connect(self.insert_completion)
        completer.setWidget(self)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer = completer
        self.textChanged.connect(self.complete)

    def insert_completion(self, completion):
        tc = self.textCursor()
        extra = len(completion) - len(self.completer.completionPrefix())
        # tc.movePosition(QTextCursor.MoveOperation.Left)
        # tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
        tc.insertText(completion[-extra:] + ", ")
        self.setTextCursor(tc)

    @property
    def text_under_cursor(self):
        tc = self.textCursor()
        tc.select(QTextCursor.SelectionType.WordUnderCursor)
        return tc.selectedText()

    def complete(self):
        prefix = self.text_under_cursor
        self.completer.setCompletionPrefix(prefix)
        popup = self.completer.popup()
        cr = self.cursorRect()
        popup.setCurrentIndex(self.completer.completionModel().index(0, 0))
        cr.setWidth(
            self.completer.popup().sizeHintForColumn(0)
            + self.completer.popup().verticalScrollBar().sizeHint().width()
        )
        self.completer.complete(cr)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.completer.popup().isVisible() and event.key() in [
            Qt.Key.Key_Enter,
            Qt.Key.Key_Return,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_Tab,
            Qt.Key.Key_Backtab,
        ]:
            event.ignore()
            return
        super().keyPressEvent(event)

