class RaceException(Exception):

    def __init__(self, message="Race Exception"):
        self.message = message
        super().__init__(message)


class QualyException(Exception):

    def __init__(self, message="Qualy Exception"):
        self.message = message
        super().__init__(message)
