import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import numpy as np


class Linear_QNet():
    def __init__(self, input_size, hidden_size, output_size, lr):
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(input_size,), name='x_input'))
        self.model.add(Dense(hidden_size, activation="relu"))
        self.model.add(Dense(output_size, activation='linear'))
        self.model.compile(loss="mean_squared_error",
                           optimizer=Adam(lr=lr),
                           metrics=["accuracy"])


    def predict(self, state):
        return self.model.predict(state)

    def predict_one(self, state):
        return self.model.predict(state[np.newaxis,...])[0]

    def __call__(self, state):
        return self.model.predict(state)

    # def forward(self, x):
    #     x = F.relu(self.linear1(x))
    #     x = self.linear2(x)
    #     return x

    def save(self, file_name='model.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        #torch.save(self.state_dict(), file_name)
        self.model.save(file_name);
        print(f"Model Saved: {file_name}")

    def load(self, file_name='model.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        self.model.load_weights(file_name)
        print(f"Model Loaded: {file_name}")


class QTrainer:
    def __init__(self, model, gamma):
        self.gamma = gamma
        self.model = model

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float)
        next_state = np.array(next_state, dtype=np.float)
        action = np.array(action, dtype=np.long)
        reward = np.array(reward, dtype=np.float)
        done = np.array(done)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            action = np.expand_dims(action, axis=0)
            reward = np.expand_dims(reward, axis=0)
            done = np.expand_dims(done, axis=0)

        # 1: predicted Q values with current state
        pred = self.model.predict(state)

        target = np.copy(pred)

        indexes = np.arange(state.shape[0])
        max_next_q = np.amax(self.model.predict(next_state), axis=1)
        action_index = np.argmax(action, axis=1)
        target[indexes, action_index[indexes]] = reward + self.gamma * np.logical_not(done) * max_next_q

        self.model.model.fit(state, target, verbose=0)



        # current_q = self._current_predict(input_states)
        # next_q = self._target_predict(next_input_states)
        # max_next_q = np.amax(next_q, axis=1)

        # target_q = np.copy(current_q)

        # indexes = np.arange(input_states.shape[0])
        # target_q[indexes, actions[indexes]] = rewards + terminal_inputs * GAMMA * max_next_q

        # result = self._model.fit(x=input_states, y=target_q, verbose=0)
