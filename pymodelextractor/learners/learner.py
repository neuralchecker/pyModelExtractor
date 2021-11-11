from abc import ABC, abstractmethod
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.teachers.teacher import Teacher


class Learner(ABC):
    @abstractmethod
    def learn(self, teacher: Teacher) -> LearningResult:
        raise NotImplementedError
