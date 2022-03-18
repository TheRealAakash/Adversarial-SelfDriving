import itertools as it
import random
import sys

sys.path.append('')
from manim import *


class NeuralNetworkMobject(VGroup):
    # Remove CONFIG since it is now deprecated in ManimCE
    # CONFIG = {
    #     "neuron_radius": 0.15,
    #     "neuron_to_neuron_buff": MED_SMALL_BUFF,
    #     "layer_to_layer_buff": LARGE_BUFF,
    #     "output_neuron_color": WHITE,
    #     "input_neuron_color": WHITE,
    #     "hidden_layer_neuron_color": WHITE,
    #     "neuron_stroke_width": 2,
    #     "neuron_fill_color": GREEN,
    #     "edge_color": LIGHT_GREY,
    #     "edge_stroke_width": 2,
    #     "edge_propogation_color": YELLOW,
    #     "edge_propogation_time": 1,
    #     "max_shown_neurons": 16,
    #     "brace_for_large_layers": True,
    #     "average_shown_activation_of_large_layer": True,
    #     "include_output_labels": False,
    #     "arrow": False,
    #     "arrow_tip_size": 0.1,
    #     "left_size": 1,
    #     "neuron_fill_opacity": 1
    # }

    # Constructor with parameters of the neurons in a list
    def __init__(
            self,
            neural_network,
            *args,

            neuron_radius=0.15,
            neuron_to_neuron_buff=MED_SMALL_BUFF,
            layer_to_layer_buff=LARGE_BUFF,
            output_neuron_color=WHITE,
            input_neuron_color=WHITE,
            hidden_layer_neuron_color=WHITE,
            neuron_stroke_width=2,
            neuron_fill_color=GREEN,
            edge_color=LIGHT_GREY,
            edge_stroke_width=2,
            edge_propogation_color=YELLOW,
            edge_propogation_time=1,
            max_shown_neurons=16,
            brace_for_large_layers=True,
            average_shown_activation_of_large_layer=True,
            include_output_labels=False,
            arrow=False,
            arrow_tip_size=0.1,
            left_size=1,
            neuron_fill_opacity=1,
            outputs=None,
            **kwargs,
    ):
        VGroup.__init__(self, *args, **kwargs)
        self.neural_network = neural_network
        self.neuron_radius = neuron_radius
        self.neuron_to_neuron_buff = neuron_to_neuron_buff
        self.layer_to_layer_buff = layer_to_layer_buff
        self.output_neuron_color = output_neuron_color
        self.input_neuron_color = input_neuron_color
        self.hidden_layer_neuron_color = hidden_layer_neuron_color
        self.neuron_stroke_width = neuron_stroke_width
        self.neuron_fill_color = neuron_fill_color
        self.edge_color = edge_color
        self.edge_stroke_width = edge_stroke_width
        self.edge_propogation_color = edge_propogation_color
        self.edge_propogation_time = edge_propogation_time
        self.max_shown_neurons = max_shown_neurons
        self.brace_for_large_layers = brace_for_large_layers
        self.average_shown_activation_of_large_layer = average_shown_activation_of_large_layer
        self.include_output_labels = include_output_labels
        self.arrow = arrow
        self.arrow_tip_size = arrow_tip_size,
        self.left_size = left_size
        self.neuron_fill_opacity = neuron_fill_opacity

        self.outputs = outputs

        self.layer_sizes = neural_network
        self.add_neurons(self.outputs)
        self.add_edges()
        self.add_to_back(self.layers)

        self.active = []

    # Helper method for constructor
    def add_neurons(self, outputs=None):
        if outputs is None:
            outputs = []
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange_submobjects(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        if self.include_output_labels:
            self.label_outputs_text(outputs)

    # Helper method for constructor

    def get_nn_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.output_neuron_color
        if index == 0:
            return self.input_neuron_color
        else:
            return self.hidden_layer_neuron_color

    # Helper method for constructor

    def get_layer(self, size, index=-1):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius=self.neuron_radius,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=BLACK,
                fill_opacity=self.neuron_fill_opacity,
            )
            for x in range(n_neurons)
        ])
        neurons.arrange_submobjects(
            DOWN, buff=self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = Tex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    # Helper method for constructor

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    # Helper method for constructor

    def get_edge(self, neuron1, neuron2, edgeColor=None):
        if edgeColor is None:
            edgeColor = self.edge_color
        if self.arrow:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.neuron_radius,
                stroke_color=edgeColor,
                stroke_width=self.edge_stroke_width,
                tip_length=self.arrow_tip_size
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=edgeColor,
            stroke_width=self.edge_stroke_width,
        )

    # Labels each input neuron with a char l or a LaTeX character
    def label_inputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            label = Tex(f"{l}_" + "{" + f"{n + 1}" + "}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each output neuron with a char l or a LaTeX character
    def label_outputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(f"{l}_" + "{" + f"{n + 1}" + "}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each neuron in the output layer with text according to an output list
    def label_outputs_text(self, outputs):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(outputs[n])
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            label.shift((neuron.get_width() + label.get_width() / 2) * RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels the hidden layers with a char l or a LaTeX character
    def label_hidden_layers(self, l):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = Tex(f"{l}_{n + 1}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)

    def activate(self, outputNum):
        # set the opacity of the neuron with index outputNum to 1
        if type(outputNum) == int:
            outputNum = [outputNum]
        transforms = []
        prevActivations = []
        for layerIndex in range(len(self.layers)):
            layer = self.layers[layerIndex]
            if len(layer.neurons) > 1:
                numToActivate = random.randint(2, len(layer.neurons) - 1)
                numToActivate = min(numToActivate, len(layer.neurons) - 1)
            else:
                numToActivate = 1
            neurons = [index for index in range(len(layer.neurons))]
            if layerIndex == len(self.layers) - 1:
                neurons = outputNum
                numToActivate = len(outputNum)
            random.shuffle(neurons)
            neurons = neurons[:numToActivate]
            neurons.sort()
            neurons = [layer.neurons[index] for index in neurons]
            for neuron in neurons:
                for neuron2 in prevActivations:
                    edge = self.get_edge(neuron2, neuron, WHITE)
                    transforms.append(Create(edge))
                    self.active.append(edge)
            for neuron in neurons:
                active = Circle(
                    radius=neuron.get_width() / 2,
                    stroke_width=self.neuron_stroke_width,
                    stroke_color=BLUE,
                    fill_color=BLUE,
                    fill_opacity=1).move_to(neuron)
                transforms.append(FadeIn(active))
                self.active.append(active)
            prevActivations = [neuron for neuron in neurons]

        return LaggedStart(*transforms)

    def reset(self):
        transforms = []
        for edge in self.active:
            transforms.append(FadeOut(edge))
        self.active = []
        return transforms
