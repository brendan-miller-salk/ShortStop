import os
import time
#TODO: make this a class that can be called from the pipeline

class PipelineProgress:
    def __init__(self):
        self.progress = "___SequenceExtractor___NegativeSet___DatabaseCombiner___FeatureExtractor___LabelPropagator"""
        self.maxSteps = len(self.progress.split("___"))+1
        self.step = 1
        self.updated = None

    def next_step(self):
        steps = self.progress.split("___")
        # print(steps)
        progress = '==='.join(steps[:self.step])
        progress += f"___{'___'.join(steps[self.step:])}"
        self.updated = progress
        if self.step < self.maxSteps:
            self.step += 1

    def run(self):
        for i in range(self.maxSteps):
            print(self.updated, end='\r')
            time.sleep(1)

            self.next_step()
        print("\n")




if __name__ == '__main__':
    pipe = PipelineProgress()
    pipe.run()