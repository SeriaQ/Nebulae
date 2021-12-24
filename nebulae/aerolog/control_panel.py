#!/usr/bin/env python
'''
control_panel
Created by Seria at 2020/10/22 10:48 AM
Email: zzqsummerai@yeah.net

                    _ooOoo_
                  o888888888o
                 o88`_ . _`88o
                 (|  0   0  |)
                 O \   。   / O
              _____/`-----‘\_____
            .’   \||  _ _  ||/   `.
            |  _ |||   |   ||| _  |
            |  |  \\       //  |  |
            |  |    \-----/    |  |
             \ .\ ___/- -\___ /. /
         ,--- /   ___\<|>/___   \ ---,
         | |:    \    \ /    /    :| |
         `\--\_    -. ___ .-    _/--/‘
   ===========  \__  NOBUG  __/  ===========
   
'''
# -*- coding:utf-8 -*-
import termios
import sys

try:
    from pynput import keyboard

    class HotKey(keyboard.HotKey):
        def __init__(self, keys, on_activate):
            super(HotKey, self).__init__(keys, on_activate)

        def press(self, key):
            """Updates the hotkey state for a pressed key.

            If the key is not currently pressed, but is the last key for the full
            combination, the activation callback will be invoked.

            Please note that the callback will only be invoked once.

            :param key: The key being pressed.
            :type key: Key or KeyCode
            """
            if key in self._keys and key not in self._state:
                self._state.add(key)
                if self._state == self._keys:
                    ret = self._on_activate()
                    return ret



    def switch_mode():
        return 0
    def show_curve_1():
        return 1
    def show_curve_2():
        return 2
    def show_curve_3():
        return 3
    def show_curve_4():
        return 4
    def show_curve_5():
        return 5
    def show_curve_6():
        return 6
    def show_curve_7():
        return 7

    def on_quit():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        return False

    class CtrlPanel(keyboard.Listener):
        """A keyboard listener supporting a number of global hotkeys.

        This is a convenience wrapper to simplify registering a number of global
        hotkeys.

        :param dict hotkeys: A mapping from hotkey description to hotkey action.
            Keys are strings passed to :meth:`HotKey.parse`.

        :raises ValueError: if any hotkey description is invalid
        """
        def __init__(self, dashboard):
            self.rank = dashboard.param['rank']
            self.db = dashboard
            self.kc = keyboard.Controller()
            if self.rank>0:
                hotkeys = {}
            else:
                hotkeys = {'<ctrl>+0': switch_mode,
                           '<ctrl>+1': show_curve_1,
                           '<ctrl>+2': show_curve_2,
                           '<ctrl>+3': show_curve_3,
                           '<ctrl>+4': show_curve_4,
                           '<ctrl>+5': show_curve_5,
                           '<ctrl>+6': show_curve_6,
                           '<ctrl>+7': show_curve_7,
                           '<backspace>': on_quit}
            self._hotkeys = [
                HotKey(HotKey.parse(key), value)
                for key, value in hotkeys.items()]
            super(CtrlPanel, self).__init__(
                on_press=self._on_press,
                on_release=self._on_release,)

        def actuate(self):
            if self.rank > 0:
                return
            fd = sys.stdin
            old = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            new[3] = new[3] & ~termios.ECHO
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, new)
                self.kc.press(keyboard.Key.backspace)
                self.kc.release(keyboard.Key.backspace)
                self.join()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        def refresh(self):
            self.kc.press(keyboard.Key.ctrl)
            self.kc.press('0')
            self.kc.release(keyboard.Key.ctrl)
            self.kc.release('0')

            self.kc.press(keyboard.Key.ctrl)
            self.kc.press('0')
            self.kc.release(keyboard.Key.ctrl)
            self.kc.release('0')

        def _on_press(self, key):
            """The press callback.

            This is automatically registered upon creation.

            :param key: The key provided by the base class.
            """
            for hotkey in self._hotkeys:
                ret = hotkey.press(self.canonical(key))
                if ret is False:
                    return ret
                elif isinstance(ret, int):
                    if ret>0:
                        self.db.panel = ret-1
                    else:
                        self.db.is_global = not self.db.is_global

        def _on_release(self, key):
            """The release callback.

            This is automatically registered upon creation.

            :param key: The key provided by the base class.
            """
            for hotkey in self._hotkeys:
                _ = hotkey.release(self.canonical(key))

except:
    CtrlPanel = None