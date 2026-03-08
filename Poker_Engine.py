import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools
import matplotlib.pyplot as plt
from collections import Counter, deque

# --- 1. БАЗОВАЯ ЛОГИКА (Card, Deck, Evaluator - без изменений) ---
class Card:
    def __init__(self, rank, suit):
        self.rank, self.suit = rank, suit
    def __repr__(self):
        r = {11:'J', 12:'Q', 13:'K', 14:'A'}.get(self.rank, str(self.rank))
        s = ['♠', '♣', '♥', '♦'][self.suit]
        return f"{r}{s}"

class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for r in range(2, 15) for s in range(4)]
        random.shuffle(self.cards)
    def deal(self, n):
        return [self.cards.pop() for _ in range(n)] if len(self.cards) >= n else []

class HandEvaluator:
    @staticmethod
    def get_score(cards):
        if len(cards) < 5: return sum([c.rank for c in cards]) / 10.0
        best_score = 0
        for combo in itertools.combinations(cards, 5):
            score = HandEvaluator._score_five(list(combo))
            if score > best_score: best_score = score
        return best_score

    @staticmethod
    def _score_five(cards):
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        counts = Counter(ranks).most_common()
        is_flush = len(set(suits)) == 1
        is_straight = len(set(ranks)) == 5 and (max(ranks) - min(ranks) == 4)
        if is_straight and is_flush: base = 800
        elif counts[0][1] == 4: base = 700
        elif counts[0][1] == 3 and counts[1][1] == 2: base = 600
        elif is_flush: base = 500
        elif is_straight: base = 400
        elif counts[0][1] == 3: base = 300
        elif counts[0][1] == 2 and counts[1][1] == 2: base = 200
        elif counts[0][1] == 2: base = 100
        else: base = 0
        return base + sum([r * (0.01 ** i) for i, r in enumerate(ranks)])

# --- 2. МОДЕЛЬ И БУФЕР ПАМЯТИ ---
class PokerLSTMNet(nn.Module):
    def __init__(self, input_size=15, hidden_size=64):
        super(PokerLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, sequence, action, reward):
        self.buffer.append((sequence, action, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

# --- 3. ИГРОК ---
class PokerEntity:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.hand = []
        self.chips = 5000
        self.current_score = 0
        self.temp_logs = [] # Текущая последовательность в раунде
        self.memory = ReplayBuffer(capacity=2000)
        self.hidden = None

    def reset_round(self):
        self.hidden = None
        self.temp_logs = []

    def decide(self, pot, board):
        state = self.encode_state(pot, board)
        state_tensor = torch.FloatTensor(state).view(1, 1, -1)
        with torch.no_grad():
            probs, self.hidden = self.model(state_tensor, self.hidden)
            action_idx = torch.multinomial(probs, 1).item()
        self.temp_logs.append(state)
        return action_idx

    def encode_state(self, pot, board):
        vec = []
        for c in self.hand: vec.extend([c.rank/14.0, (c.suit+1)/4.0])
        tmp_b = board + [None]*(5-len(board))
        for c in tmp_b:
            if c: vec.extend([c.rank/14.0, (c.suit+1)/4.0])
            else: vec.extend([0.0, 0.0])
        vec.append(pot/20000.0)
        return vec

# --- 4. ОБУЧЕНИЕ И СИМУЛЯЦИЯ ---
def train_with_replay():
    models = [PokerLSTMNet() for _ in range(3)]
    optimizers = [optim.Adam(m.parameters(), lr=0.0003) for m in models]
    
    # Чтобы боты догоняли, дадим им Player_0 и Player_1
    players = [PokerEntity(f"Bot_{i}", models[i]) for i in range(2)]
    players.append(PokerEntity("Pivot", models[2]))
    
    history_pivot = []
    
    print("--- Запуск войны ИИ: LSTM + Experience Replay ---")

    for i in range(1, 1501):
        # Раунд игры
        deck = Deck()
        board, pot = [], 0
        for p in players:
            p.hand = deck.deal(2)
            p.reset_round()

        actions_taken = [] # Для логов
        for n in [0, 3, 1, 1]:
            board.extend(deck.deal(n))
            for p in players:
                action = p.decide(pot, board)
                bet = 200 if action == 2 else (100 if action == 1 else 0)
                p.chips -= bet
                pot += bet
                p.current_score = HandEvaluator.get_score(p.hand + board)
                actions_taken.append(action)

        winner_idx = [p.current_score for p in players].index(max([p.current_score for p in players]))
        players[winner_idx].chips += pot

        # Сохранение опыта в буфер
        for p_idx, p in enumerate(players):
            reward = 1.0 if p_idx == winner_idx else -0.5
            # Сохраняем всю цепочку состояний раунда и последнее действие
            p.memory.push(p.temp_logs, actions_taken[p_idx], reward)

        # Обучение на случайных батчах из памяти
        for p_idx, p in enumerate(players):
            batch = p.memory.sample(32)
            if not batch: continue
            
            total_loss = 0
            for seq, act, rew in batch:
                st_tensor = torch.FloatTensor(seq).unsqueeze(0)
                probs, _ = p.model(st_tensor)
                
                # Защита от выхода за границы, если действий было меньше
                act_idx = min(act, 2) 
                chosen_prob = probs[0][act_idx]
                
                loss = -(torch.log(chosen_prob + 1e-8) * rew)
                total_loss += loss

            optimizers[p_idx].zero_grad()
            (total_loss / len(batch)).backward()
            optimizers[p_idx].step()

        history_pivot.append(players[2].chips)

        if i % 250 == 0:
            print(f"Партия {i:4} | Баланс Pivot: {players[2].chips} | Память Бота_0: {len(players[0].memory.buffer)}")

    plt.figure(figsize=(12, 6))
    plt.plot(history_pivot, label='Pivot Balance')
    plt.axhline(y=5000, color='r', linestyle='--')
    plt.title('Борьба интеллектов (LSTM + Replay Buffer)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_with_replay()