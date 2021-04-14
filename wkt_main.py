import random

from data.E2E.reader import E2EDataReader
from data.WebNLG.reader import WebNLGDataReader
from data.WKT.reader import WTQAnnotationsDataReader
from data.reader import DataReader, DataSetType
from planner.naive_planner import NaivePlanner
from planner.neural_planner import NeuralPlanner
from planner.planner import Planner
from process.evaluation import EvaluationPipeline
from process.pre_process import TrainingPreProcessPipeline, TestingPreProcessPipeline
from process.reg import REGPipeline
from process.train_model import TrainModelPipeline
from process.train_planner import TrainPlannerPipeline
from process.translate import TranslatePipeline
from reg.bert import BertREG
from reg.naive import NaiveREG
from reg.base import REG
from scorer.global_direction import GlobalDirectionExpert
from scorer.product_of_experts import WeightedProductOfExperts
from scorer.relation_direction import RelationDirectionExpert
from scorer.relation_transitions import RelationTransitionsExpert
from scorer.splitting_tendencies import SplittingTendenciesExpert
from utils.pipeline import Pipeline


class Config:
    def __init__(self, reader: DataReader = None, planner: Planner = None, reg: REG = None, test_reader: DataReader = None, low_mem: bool = None):
        self.reader = {
            DataSetType.TRAIN: reader,
            DataSetType.DEV: reader,
            DataSetType.TEST: test_reader if test_reader else reader,
        }
        self.planner = planner
        self.reg = reg
        self.low_mem = low_mem



if __name__ == "__main__":
    # naive_planner = NaivePlanner(WeightedProductOfExperts([
    #     RelationDirectionExpert,
    #     GlobalDirectionExpert,
    #     SplittingTendenciesExpert,
    #     RelationTransitionsExpert
    # ]))

    naive_planner = NaivePlanner(WeightedProductOfExperts([
        RelationDirectionExpert,
        GlobalDirectionExpert,
        SplittingTendenciesExpert,
        RelationTransitionsExpert
    ]))
    neural_planner = NeuralPlanner()
    # combined_planner = CombinedPlanner((neural_planner, naive_planner))
    config2 = Config(reader=WebNLGDataReader,
                    planner=naive_planner,
                    reg=BertREG, low_mem=True)
    NaivePipeline = Pipeline()
    NaivePipeline.enqueue("pre-process", "Pre-process training data", TrainingPreProcessPipeline)
    NaivePipeline.enqueue("train-planner", "Train Planner", TrainPlannerPipeline)
    NaivePipeline.enqueue("train-model", "Train Model", TrainModelPipeline)
    NaivePipeline.enqueue("train-reg", "Train Referring Expressions Generator", REGPipeline)
    #res = NaivePipeline.mutate({"config": config}).execute("WTQtest", cache_name="WTQexp_naive")
    #res = MainPipeline.mutate({"config": config}).execute("WTQtest", cache_name="WTQexp_neural")
    #res = MainPipeline.mutate({"config": config}).execute("WTQtest", cache_name="WTQexp_naive")
    config2 = Config(reader=WTQAnnotationsDataReader,planner=naive_planner,
                    reg=BertREG, low_mem=True)
    
    test = TestingPreProcessPipeline.mutate({"config": config2})
    NaivePipeline.enqueue("test-corpus", "Pre-process test data", test)
    NaivePipeline.enqueue("translate", "Translate Test", TranslatePipeline)
    NaivePipeline.enqueue("evaluate", "Evaluate Translations", EvaluationPipeline)
    #following line doesn't work, cos res is a CacheDict, with cache_dict, and val_dict as attrs, val_dict contains "config"
    #res1 = NaivePipeline.mutate(res.update({"config": config})).execute("WTQtest", cache_name="WTQexp_naive")
    
    config1 = Config(reader=WebNLGDataReader,
                    planner=neural_planner,
                    reg=BertREG, low_mem=False)
    NeuralPipeline = Pipeline()
    #MainPipeline = Pipeline()
    NeuralPipeline.enqueue("pre-process", "Pre-process training data", TrainingPreProcessPipeline)
    NeuralPipeline.enqueue("train-planner", "Train Planner", TrainPlannerPipeline)
    NeuralPipeline.enqueue("train-model", "Train Model", TrainModelPipeline)
    NeuralPipeline.enqueue("train-reg", "Train Referring Expressions Generator", REGPipeline)
    #res = MainPipeline.mutate({"config": config}).execute("WTQtest", cache_name="WTQexp_neural")
    #res = MainPipeline.mutate({"config": config}).execute("WTQtest", cache_name="WTQexp_naive")
    config1 = Config(reader=WTQAnnotationsDataReader,planner=neural_planner,
                    reg=BertREG, low_mem=False)
 
    test = TestingPreProcessPipeline.mutate({"config": config1})
    NeuralPipeline.enqueue("test-corpus", "Pre-process test data", test)
    NeuralPipeline.enqueue("translate", "Translate Test", TranslatePipeline)
    NeuralPipeline.enqueue("evaluate", "Evaluate Translations", EvaluationPipeline)
    
    #MainPipeline.enqueue("test-corpus", "Pre-process test data", test)
    #res['evaluate'] = r['evaluate']
    #ExperimentsPipeline = Pipeline()

    #res2 = NaivePipeline.mutate({"config": config2}).execute("WTQtest", cache_name="WTQexp_naive")
    res1 = NeuralPipeline.mutate({"config": config1}).execute("WTQtest", cache_name="WTQexp_neural")

    print()

    d = random.choice(res1["translate"].data)
    print("Random Sample:")
    print("Graph:", d.graph.as_rdf())
    print("Plan:", d.plan)
    print("Translation:", d.hyp)
    print("Reference:  ", d.text)

    print()

    #print("Naive BLEU", res2["evaluate"]["bleu"])
    print("Neural BLEU", res1["evaluate"]["bleu"])
