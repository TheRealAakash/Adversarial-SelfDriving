# Citation: Box Of Hats (https://github.com/Box-Of-Hats )
# avail: https://raw.githubusercontent.com/Sentdex/pygta5/master/getkeys.py

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def removeAll(the_list, val):
    return [value for value in the_list if value != val]


class KeyChecker:
    def __init__(self):
        self.justPressed = []
        self.alreadyPressed = keyList[:]

    def update(self):
        self.justPressed = list(set(self.justPressed))
        self.alreadyPressed = list(set(self.alreadyPressed))
        key_check()
        keys = key_check()
        for key in keyList:
            if key in self.alreadyPressed:
                if key in self.justPressed:
                    self.justPressed = removeAll(self.justPressed, key)
            if key in keys:
                if key not in self.alreadyPressed:
                    self.justPressed.append(key)
            else:
                if key in self.alreadyPressed:
                    self.alreadyPressed = removeAll(self.alreadyPressed, key)

    def checkKey(self, key):
        self.update()
        if key in self.justPressed:
            if key not in self.alreadyPressed:
                self.alreadyPressed.append(key)
            return True
        else:
            return False
