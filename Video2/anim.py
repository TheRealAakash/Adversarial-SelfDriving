import itertools as it
import random
import sys
from manim import *
from NeuralNetwork import NeuralNetworkMobject

sys.path.append('..')

from datasets import load_datasets

config.disable_caching = True
config.write_all = False
(trainX, trainY), (_, _), (_, _), _ = load_datasets.load_data_traffic_signs()
stopSigns = []
for x, y in zip(trainX, trainY):
    if y == 14:
        stopSigns.append(x)


class AdversarialSelfDriving(MovingCameraScene):
    def construct(self):
        car = ImageMobject("images/tesla.png").scale(0.8).move_to(LEFT * 3)
        self.play(FadeIn(car, shift=RIGHT))
        CNN = NeuralNetworkMobject(neural_network=[4, 5, 6, 3], include_output_labels=True, outputs=["Stop Sign", "Speed limit 30", "Speed limit 60"])
        self.play(car.animate.move_to(RIGHT * 4), run_time=4)
        self.play(FadeOut(car), run_time=2)
        self.play(Create(CNN), run_time=3)
        mph60Sign = ImageMobject("images/60mphsign.png").scale(1).move_to(LEFT * 3.5)
        self.wait(4)
        label = Text("Convolutional Neural Networks(CNN)").scale(1).move_to(UP * 3.5)
        self.play(Create(label))
        self.play(FadeIn(mph60Sign), run_time=1)
        self.play(FadeOut(mph60Sign, shift=RIGHT, run_time=1), CNN.activate(2), run_time=2)
        self.play(*CNN.reset())
        numTrials = 30
        for ind in range(1, numTrials):
            p = [ind * ind, numTrials / 5, numTrials / 5]
            choice = np.random.choice([0, 1, 2], p=[n / sum(p) for n in p])
            choice = int(choice)
            sign = ImageMobject(stopSigns[ind]).scale(3).move_to(LEFT * 3)
            sleepTime = 1 / ind
            self.play(FadeIn(sign, run_time=sleepTime))
            self.play(FadeOut(sign, shift=RIGHT), CNN.activate(choice), run_time=sleepTime)
            self.play(*CNN.reset(), run_time=sleepTime)
        self.play(FadeOut(CNN))
        # 5 by 10 grid of stop signs
        signs = Group()
        signsToRemove = []
        signsKept = []
        self.play(Uncreate(label))
        for i in range(5):
            row = Group()
            for j in range(8):
                sign = ImageMobject(stopSigns[i * 5 + j]).scale(2).move_to(LEFT * 3)
                row.add(sign)
                if random.random() < 0.3:
                    signsToRemove.append(sign)
                else:
                    signsKept.append(sign)
            row.arrange_submobjects(RIGHT, buff=0.5)
            signs.add(row)
        signs.arrange_submobjects(DOWN, buff=0.5)
        signs.shuffle_submobjects()
        self.play(FadeIn(signs))
        self.wait(2)
        random.shuffle(signsToRemove)
        for sign in signsToRemove:
            signs.remove(sign)
        self.play(LaggedStart(*[FadeOut(sign) for sign in signsToRemove]), run_time=6)
        self.play(*[FadeOut(sign) for sign in signsKept])
        self.play(FadeIn(CNN))

        stopSignPerturbed = ImageMobject("images/StopSignPerturbed.png").scale(3).move_to(LEFT * 4.3)
        stopSign = ImageMobject("images/StopSign.png").scale(3).move_to(LEFT * 3)
        perturbedText = Text("Perturbed").scale(0.5).move_to(stopSignPerturbed, DOWN)
        self.play(FadeIn(stopSign), FadeIn(stopSignPerturbed), FadeIn(perturbedText))
        self.play(FadeOut(stopSign, shift=RIGHT), stopSignPerturbed.animate.shift(RIGHT), perturbedText.animate.shift(RIGHT), CNN.activate(0))
        self.play(*CNN.reset())
        self.play(FadeOut(stopSignPerturbed, shift=RIGHT), FadeOut(perturbedText, shift=RIGHT), CNN.activate(1))
        self.play(*CNN.reset())
        self.play(CNN.animate.move_to(RIGHT * 4.7).scale(0.6))
        CNNText = Text("Classifier").scale(0.5).move_to(CNN, DOWN * 2).shift(DOWN * 0.5 + LEFT * 0.1)
        self.play(Create(CNNText))

        generator = NeuralNetworkMobject([4, 5, 3, 5, 4]).scale(0.6).shift(LEFT * 4.7)
        generatorText = Text("Generator").scale(0.5).move_to(generator, DOWN).shift(DOWN * 0.5)
        self.play(Create(generator), Create(generatorText))

        discriminator = NeuralNetworkMobject([4, 3, 1])
        discriminatorText = Text("Discriminator").scale(0.5).move_to(discriminator, DOWN * 2).shift(DOWN * 0.5)
        self.play(Create(discriminator), Create(discriminatorText))

        titleText = Text("Adversarial Self-Driving Framework").scale(.7).move_to(UP * 3)
        self.play(Create(titleText))

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(CNN).shift(LEFT * 0.1).set(width=CNN.width * 1.6))
        stopSign.move_to(CNN, LEFT).shift(LEFT * 1.2)
        self.play(FadeIn(stopSign.scale(0.6)))
        self.play(FadeOut(stopSign, shift=RIGHT), CNN.activate(0))

        # self.play(*CNN.reset())

        self.play(self.camera.frame.animate.move_to(generator).set(width=generator.width * 1.6))
        generatorSign = ImageMobject("images/StopSign.png").scale(1.3).move_to(generator, LEFT * 1.2).shift(LEFT)
        self.play(FadeIn(generatorSign))
        self.play(FadeOut(generatorSign, shift=RIGHT), generator.activate([0, 1, 2, 3]))
        stopSignPerturbed.move_to(generator, RIGHT).shift(RIGHT)
        stopSignPerturbed.scale(0.6)
        self.play(FadeIn(stopSignPerturbed, shift=RIGHT * 0.9))

        self.play(self.camera.frame.animate.move_to(discriminator).set(width=discriminator.width * 2))
        self.play(stopSignPerturbed.animate.shift(RIGHT * 0.8))
        self.play(FadeOut(stopSignPerturbed, shift=RIGHT), discriminator.activate(0))
        discriminatorOutput = Text("0.13").scale(0.5).move_to(discriminator, RIGHT).shift(RIGHT * 0.8)
        self.play(FadeIn(discriminatorOutput))

        self.wait(2)
        self.play(Restore(self.camera.frame))
