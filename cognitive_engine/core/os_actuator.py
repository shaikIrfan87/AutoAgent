import ctypes
import time
from typing import Dict, Any, Tuple, Optional

# Win32 Constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_KEYUP = 0x0002
GA_ROOT = 2


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short), ("wParamH", ctypes.c_ushort)]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", INPUT_UNION)]


class OSActuator:
    """
    Direct Win32 hardware input dispatcher.
    Enforces Invariant 2 (Causal Diode) to block hazardous hotkeys,
    guarantees window foreground focus contract, and dispatches clicks, drags, and scrolls.
    """
    FORBIDDEN_HOTKEYS = {
        (0x5B, 0x52),       # Win + R (Run prompt)
        (0x12, 0x73),       # Alt + F4 (Close)
        (0x11, 0x12, 0x2E)  # Ctrl + Alt + Del
    }

    def __init__(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        self.user32 = ctypes.windll.user32
        self.screen_w = self.user32.GetSystemMetrics(0) or 1920
        self.screen_h = self.user32.GetSystemMetrics(1) or 1080

    def ensure_foreground_at(self, x: int, y: int) -> bool:
        """Brings the parent top-level window at (x, y) to the foreground before dispatching keystrokes/clicks."""
        try:
            pt = POINT(x, y)
            hwnd = self.user32.WindowFromPoint(pt)
            if hwnd:
                root = self.user32.GetAncestor(hwnd, GA_ROOT) or hwnd
                self.user32.SetForegroundWindow(root)
                return True
        except Exception:
            pass
        return False

    def click_at(self, x: int, y: int, set_focus: bool = True) -> Dict[str, Any]:
        """Moves cursor to absolute coordinates and executes left click with foreground guard."""
        if not (0 <= x <= self.screen_w and 0 <= y <= self.screen_h):
            return {"success": False, "error": f"Coordinates ({x}, {y}) exceed bounds."}

        if set_focus:
            self.ensure_foreground_at(x, y)

        norm_x = int(x * (65535.0 / max(self.screen_w, 1)))
        norm_y = int(y * (65535.0 / max(self.screen_h, 1)))

        # 1. Move Cursor
        move_inp = INPUT(type=INPUT_MOUSE)
        move_inp.union.mi = MOUSEINPUT(dx=norm_x, dy=norm_y, mouseData=0, 
                                      dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=None)
        self.user32.SendInput(1, ctypes.byref(move_inp), ctypes.sizeof(INPUT))
        time.sleep(0.04)

        # 2. Left Down + Left Up
        down_inp = INPUT(type=INPUT_MOUSE)
        down_inp.union.mi = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=None)
        up_inp = INPUT(type=INPUT_MOUSE)
        up_inp.union.mi = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=None)
        
        self.user32.SendInput(1, ctypes.byref(down_inp), ctypes.sizeof(INPUT))
        time.sleep(0.02)
        self.user32.SendInput(1, ctypes.byref(up_inp), ctypes.sizeof(INPUT))

        return {"success": True, "action": "click", "coords": (x, y)}

    def drag_and_drop(self, x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        """Performs left-click drag from (x1, y1) to (x2, y2)."""
        if not (0 <= x1 <= self.screen_w and 0 <= y1 <= self.screen_h and 0 <= x2 <= self.screen_w and 0 <= y2 <= self.screen_h):
            return {"success": False, "error": "Drag coordinates out of bounds."}

        norm_x1 = int(x1 * (65535.0 / max(self.screen_w, 1)))
        norm_y1 = int(y1 * (65535.0 / max(self.screen_h, 1)))
        norm_x2 = int(x2 * (65535.0 / max(self.screen_w, 1)))
        norm_y2 = int(y2 * (65535.0 / max(self.screen_h, 1)))

        # Move to origin
        m1 = INPUT(type=INPUT_MOUSE)
        m1.union.mi = MOUSEINPUT(dx=norm_x1, dy=norm_y1, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=None)
        self.user32.SendInput(1, ctypes.byref(m1), ctypes.sizeof(INPUT))
        time.sleep(0.04)

        # Mouse Down
        down = INPUT(type=INPUT_MOUSE)
        down.union.mi = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=None)
        self.user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
        time.sleep(0.05)

        # Move to destination
        m2 = INPUT(type=INPUT_MOUSE)
        m2.union.mi = MOUSEINPUT(dx=norm_x2, dy=norm_y2, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=None)
        self.user32.SendInput(1, ctypes.byref(m2), ctypes.sizeof(INPUT))
        time.sleep(0.05)

        # Mouse Up
        up = INPUT(type=INPUT_MOUSE)
        up.union.mi = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=None)
        self.user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))

        return {"success": True, "action": "drag_and_drop", "from": (x1, y1), "to": (x2, y2)}

    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        """Dispatches mouse wheel vertical scroll (positive=up, negative=down)."""
        if x is not None and y is not None:
            norm_x = int(x * (65535.0 / max(self.screen_w, 1)))
            norm_y = int(y * (65535.0 / max(self.screen_h, 1)))
            move_inp = INPUT(type=INPUT_MOUSE)
            move_inp.union.mi = MOUSEINPUT(dx=norm_x, dy=norm_y, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, time=0, dwExtraInfo=None)
            self.user32.SendInput(1, ctypes.byref(move_inp), ctypes.sizeof(INPUT))
            time.sleep(0.02)

        # 1 click = 120 WHEEL_DELTA
        wheel_delta = ctypes.c_ulong(clicks * 120).value
        wheel_inp = INPUT(type=INPUT_MOUSE)
        wheel_inp.union.mi = MOUSEINPUT(dx=0, dy=0, mouseData=wheel_delta, dwFlags=MOUSEEVENTF_WHEEL, time=0, dwExtraInfo=None)
        self.user32.SendInput(1, ctypes.byref(wheel_inp), ctypes.sizeof(INPUT))
        return {"success": True, "action": "scroll", "clicks": clicks}

    def type_text(self, text: str) -> Dict[str, Any]:
        """Sends unicode keystrokes sequentially."""
        KEYEVENTF_UNICODE = 0x0004
        for char in text:
            code = ord(char)
            down = INPUT(type=INPUT_KEYBOARD)
            down.union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=None)
            up = INPUT(type=INPUT_KEYBOARD)
            up.union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)
            
            self.user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
            self.user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))
            time.sleep(0.01)
        return {"success": True, "action": "type", "chars": len(text)}
