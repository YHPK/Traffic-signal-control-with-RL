import matplotlib.pyplot as plt

def plot(data, name):
    plt.figure(figsize=(16,8))
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.plot(data)
    plt.savefig(f'{name}.png')