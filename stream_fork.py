
class StreamFork:

    def __init__(self, *streams):
        self.streams = list(set(list(streams)))

    def write(self, __s):
        for stream in self.streams:
            stream.write(__s)

    def close(self):
        for stream in self.streams:
            stream.close()
