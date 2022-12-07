from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator

from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner
from pymodelextractor.learners.observation_table_learners.pdfa_lstarcol_learner import PDFALStarColLearner
from pymodelextractor.learners.observation_table_learners.translators.pdfa_lstar_quant_observation_table_translator import \
    PDFALStarQuantObservationTableTranslation
from pymodelextractor.learners.observation_table_learners.translators.pdfa_lstarcol_quant_observation_table_translator import \
    PDFALStarColQuantObservationTableTranslator
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pymodelextractor.teachers.pdfa_teacher import PDFATeacher


class PDFAExtractionFactory:

    def __init__(self) -> None:
        super().__init__()

    def probabilistic_lstar_tolerance_extraction(self, model: WeightedAutomaton, tolerance):
        comparator = WFAToleranceComparator(tolerance)
        learner = PDFALStarLearner(comparator)
        teacher = PDFATeacher(model, comparator)
        return learner, teacher

    def probabilistic_lstarcol_tolerance_extraction(self, model: WeightedAutomaton, tolerance):
        comparator = WFAToleranceComparator(tolerance)
        learner = PDFALStarColLearner(comparator)
        teacher = PDFATeacher(model, comparator)
        return learner, teacher

    def probabilistic_lstar_quant_extraction(self, model: WeightedAutomaton, partitions):
        comparator = WFAQuantizationComparator(partitions)
        translator = PDFALStarQuantObservationTableTranslation()
        learner = PDFALStarLearner(comparator, translator)
        teacher = PDFATeacher(model, comparator)
        return learner, teacher

    def probabilistic_lstarcol_quant_extraction(self, model: WeightedAutomaton, partitions):
        comparator = WFAQuantizationComparator(partitions)
        translator = PDFALStarColQuantObservationTableTranslator()
        learner = PDFALStarColLearner(comparator, translator)
        teacher = PDFATeacher(model, comparator)
        return learner, teacher

    def pac_lstar_tolerance_extraction(self, model: ProbabilisticModel, tolerance):
        comparator = WFAToleranceComparator(tolerance)
        learner = PDFALStarLearner(comparator)
        teacher = PACProbabilisticTeacher(model, comparator)
        return learner, teacher

    def pac_lstar_quant_extraction(self, model: ProbabilisticModel, partitions):
        comparator = WFAQuantizationComparator(partitions)
        translator = PDFALStarQuantObservationTableTranslation()
        learner = PDFALStarLearner(comparator, translator)
        teacher = PACProbabilisticTeacher(model, comparator)
        return learner, teacher

