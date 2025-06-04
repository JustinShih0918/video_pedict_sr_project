class DebugFlag:
    def __init__(self):
        self.flag = False

    def set_flag(self, value):
        self.flag = value

    def is_enabled(self):
        return self.flag

debug_flag = DebugFlag()

def set_flag(value):
    debug_flag.set_flag(value)

def debug(message):
    if debug_flag.is_enabled():
        print(message)