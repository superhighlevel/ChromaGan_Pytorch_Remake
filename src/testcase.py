from trainer import Tester


class TestCase:
    def __init__(self):
        self.tester = Tester()

    def test(self):
        self.tester.test()
        print('Testing done!')
