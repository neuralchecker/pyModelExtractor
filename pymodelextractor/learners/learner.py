from abc import ABC, abstractmethod
from learners.learning_result import LearningResult
from teachers.teacher import Teacher


class Learner(ABC):
    @abstractmethod
    def learn(self, teacher: Teacher) -> LearningResult:
        raise NotImplementedError
