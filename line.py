# the class for the line which is the observable 
# notify system if there is a change so the system can change volume
import mediapipe as mp
from point import Point

class Observable:
    #Observable) manages observers and notifies them of change
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    def notify(self):
        #Notify observers
        for observer in self._observers:
            observer.update(self)

class Line(Observable):
    point1: Point
    point2: Point
    right_hand: bool

    def __init__(self, point1: Point, point2: Point, right: bool) -> None:
        self.point1 = point1
        self.point2 = point2
        self.right_hand = right
    
    def change_line(self, point1:Point, point2:Point) -> None:
        self.point1 = point1
        self.point2 = point2
        self.notify()
