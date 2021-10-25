import os

import cv2
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.layers import GlobalAveragePooling2D

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # -1:cpu, 0:first gpu
import pylab
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter

# tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
from tqdm import tqdm
import keyboard

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


def buildModel(input_shape, action_space):
    # X_input = Input(input_shape)
    # model = Flatten()(model)
    # model = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(model)
    base_model = Xception(weights=None, include_top=False, input_shape=input_shape)
    model = base_model.output
    model = GlobalAveragePooling2D()(model)
    predictions = Dense(action_space, activation="linear")(model)
    # output = Dense(action_space, activation="tanh")(model)

    actor = Model(inputs=base_model.input, outputs=predictions)
    actor.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    return actor


class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env, env_name, pixel_based, trainBatch, model_name=""):
        # Initialization
        # Environment and PPO parameters

        self.pixels = pixel_based
        self.env_name = env_name
        self.env = env
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 2000000  # total episodes to train through all environments
        self.episode = 0  # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0  # when average score is above 0 model will be saved
        self.learning_rate = 0.00025
        self.epochs = 5  # training epochs
        self.shuffle = True
        self.trainingBatch = trainBatch
        # self.optimizer = RMSprop
        self.optimizer = Adam

        self.replay_count = 0
        self.writer = SummaryWriter(comment="_" + self.env_name + "_" + self.optimizer.__name__ + "_" + str(self.learning_rate))

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], []  # used in matplotlib plots

        # Create actor-Critic network models
        self.actor = buildModel(self.state_size, self.action_size)

        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"
        # self.load() # uncomment to continue training from old weights
        pylab.figure(figsize=(18, 9))
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)
        self.framesTrained = np.load("frames.npy")

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.actor.predict(state)

        low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)

        return action

    def train(self, states, actions, next_states):
        # y_true = np.array(actions)
        self.framesTrained += len(states)
        states = np.vstack(states)
        actions = np.vstack(actions)
        states = states.reshape(-1, *self.state_size)

        self.actor.fit(states, actions, epochs=self.epochs, batch_size=128, verbose=1, shuffle=self.shuffle)
        self.save()
        np.save("frames.npy", self.framesTrained)
        print(f"Trained on {self.framesTrained} frames...")
        self.env.reset()

    def load(self, actor_name="", critic_name=""):
        if actor_name:
            self.actor.load_weights(actor_name)
        else:
            self.actor.load_weights(self.Actor_name)

    def save(self):
        self.actor.save_weights(self.Actor_name)

    def processState(self, state):
        if self.pixels:
            state = np.array([state])
        else:
            state = np.reshape(state, [1, self.state_size[0]])
        return state

    def run_batch(self):
        state = self.env.reset()
        state = self.processState(state)
        done, score, SAVING = False, 0, ''
        self.env.autopilot(True)
        while True:
            # Instantiate or reset games memory
            states, next_states, actions = [], [], []
            print()
            for t in tqdm(range(self.trainingBatch)):
                # actor picks an action
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, action = self.env.step([1, 0, 0])
                self.env.render()
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states.append(state)
                next_states.append(self.processState(next_state))
                actions.append(action)
                # Update current state shape
                state = self.processState(next_state)
                score += reward
                if keyboard.is_pressed("p"):
                    print()
                    self.test(1, False)
                    self.env.reset()
                    print()

                if done:
                    self.episode += 1
                    # print(f"episode: {self.episode}/{self.EPISODES}, score: {score}, average: {average:.2f} {SAVING}")
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.learning_rate, self.episode)
                    # self.writer.add_scalar(f'Workers:{1}/average_score', average, self.episode)

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = self.processState(state)
            print("")
            self.train(states, actions, next_states)
            if self.episode >= self.EPISODES:
                break

        self.env.close()

    def test(self, test_episodes=100, closeAfter=True):  # evaluate
        print("Testing...")
        # self.load()
        for e in range(test_episodes):
            state = self.env.reset()
            self.env.autopilot(False)
            state = self.processState(state)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = self.actor.predict(state)[0]
                state, reward, done, _ = self.env.step(action)
                state = self.processState(state)
                score += reward
                if keyboard.is_pressed("b"):
                    print(action)
                if done:
                    # average, SAVING = self.plotModel(score, e, save=False)
                    # print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                    break
        if closeAfter:
            self.env.close()
        print("Testing stopped...")
        self.env.autopilot(True)

    def testPerturbed(self, generator, numEpisodes=5):
        x_train, y_train = [], []
        print("Testing perturbed...")
        self.load()
        for e in range(numEpisodes):
            state = self.env.reset()
            state = self.processState(state)
            done = False
            score = 0
            while not done:
                self.env.render()
                # cv2.waitKey(1)
                action = self.actor.predict(state)[0]
                state, reward, done, _ = self.env.step(action)
                x_train.append(state)
                y_train.append(action)
                if keyboard.is_pressed("b"):
                    print(action)
                    # self.env.world.hud.notification("Traffic light changed! Good to go!")
                state = generator.predict(np.array([state]))[0]
                cv2.imshow("perturbed", cv2.resize(np.array(state * 255, np.uint8), dsize=(256, 256)))
                state = self.processState(state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e + 1, numEpisodes, score))
                    break
        self.env.close()
        return x_train, y_train

    def generate(self, numFrames=100, closeAfter=True):  # evaluate
        print("Generating...")
        # self.load()
        x_data, y_data = [], []
        state = self.env.reset()
        self.env.autopilot(False)
        state = self.processState(state)
        done = False
        # score = 0
        for f in tqdm(range(numFrames)):
            self.env.render()
            action = self.actor.predict(state)[0]
            state, reward, done, _ = self.env.step(action)
            if keyboard.is_pressed("b"):
                print(action)
            y_data.append(action)
            x_data.append(state)
            state = self.processState(state)
            # score += reward
            if done:
                # print("frames: {}/{}, score: {}, average{}".format(f, numFrames, score, average))
                state = self.env.reset()
                state = self.processState(state)
                done = False
                score = 0

        if closeAfter:
            self.env.close()
        print("Testing stopped...")
        return x_data, y_data


def main(enableTrain=False):
    print("starting...")
    import carlaEnv
    env_name = 'CarEnv'
    env = carlaEnv.CarEnv(True)
    try:
        agent = PPOAgent(env, env_name, True, 5_000)
        try:
            agent.load()
            print("Model loaded...")
        except:
            pass
        if enableTrain:
            agent.run_batch()  # train as PPO
        env.autopilot(False)
        agent.test(10)
    # except (Exception, InterruptedError, KeyboardInterrupt) as e:
    except (Exception, InterruptedError, KeyboardInterrupt) as e:
        print(e)
        # print("Interrupted")
        env.close()
        try:
            print("Press control c now")
            time.sleep(3)
            print("Don't press control c")
        except:
            print("Quitting...")
            return
    main(enableTrain=enableTrain)


if __name__ == "__main__":
    main()
    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
