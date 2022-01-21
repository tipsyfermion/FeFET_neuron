
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"src"))

import matplotlib.pyplot as plt

from neuron import Neuron
def main():
    HNN = Neuron(os.path.join(os.getcwd(),"config.json"))
    plt.plot(HNN.sweep(HNN.alpha_n,-70,20,0.1))
    plt.plot(25/0.1,HNN.alpha_n(25.0),"r.")
    plt.show()
    plt.plot(HNN.sweep(HNN.alpha_m,-70,20,0.1))
    plt.show()

    plt.plot(HNN.sweep(HNN.alpha_h,-70,20,0.1))
    plt.show()

    plt.plot(HNN.sweep(HNN.beta_n,-70,20,0.1))
    plt.show()

    plt.plot(HNN.sweep(HNN.beta_m,-70,20,0.1))
    plt.show()

    plt.plot(HNN.sweep(HNN.beta_h,-70,20,0.1))  
    plt.show()
    v, n, m, h, time = HNN.simulate(1000)
    
    plt.plot(time,v)
    # plt.show()
    # plt.plot(time, n)
    # plt.show()
    # plt.plot(time, m)
    # plt.show()
    # plt.plot(time, h)
    # plt.show()
    return

if __name__ == "__main__":
    main()