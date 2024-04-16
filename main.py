import numpy as np

def gpt(input: list[int]) -> list[list[float]]:
    # inputs は [n_seq] の形状を持つ
    # 出力 [n_seq, n_vocab] の形状を持つ
    output = # neural network を通して決まる
    return output

# 自己回帰の実装
def generate(inputs, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate): # 自己回帰的デコードループ
        output = gpt(inputs) # モデルのフォワードパス
        next_id = np.argmax(output[-1]) # 貪欲サンプリング
        inputs.append(int(next_id)) # 予測を入力に追加
    return inputs[len(inputs) - n_tokens_to_generate :]
# 生成されたIDのみを返す

def lm_loss(inputs: list[int], params) -> float:
    x, y = inputs[:-1], inputs[1:]

    # フォワードパス
    output = gpt(x, params)

    #クロスエントロピー損失
    loss = np.mean(-np.log(output[y]))

    return loss

def train(texts: list[list[str]], params) -> float:
    for text in texts:
        inputs = tokenizer.encode(text)
        loss = lm_loss(inputs, params)
        gradients = compute_gradients_via_backpropagation(loss, params)
        params = gradient_descent_update_step(gradients, params)
    return params

vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]

input_ids = [1, 0] # "not" "all"
output_ids = generate(input_ids, 3) # output_ids = [2, 4, 6]
output_tokens = [vocab[i] for i in output_ids]

inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear"
output = gpt(inputs)
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes