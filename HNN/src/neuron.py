import numpy as np
import sympy as sym
import json


class Neuron():
    def __init__(self, conf_path) -> None:        
        with open(conf_path) as file:
            self.conf = json.load(file)["HHN"]
            
        self.init_equations()

        self.g_Na = self.conf["g_Na"]
        self.g_K = self.conf["g_K"]
        self.g_L = self.conf["g_L"]
        
        self.E_Na = self.conf["E_Na"]
        self.E_K = self.conf["E_K"]
        self.E_L = self.conf["E_L"]
        
        self.C = self.conf["C"]
        
        self.n = 0
        self.m = 0
        self.h = 0.5
        self.v = -63
        self.dt =self.conf["dt"]
        return
    
    def simulate(self, time):
        
        nt = int(time/self.dt)
        
        v = np.zeros(nt)
        n = np.zeros(nt)
        m = np.zeros(nt)
        h = np.zeros(nt)
        time = np.zeros(nt)
        
        for t in range(nt):
            self.step()
            v[t] = self.v
            n[t] = self.n
            m[t] = self.m
            h[t] = self.h
            time[t] = t*self.dt
        
        return v, n, m, h, time
    
    def step(self, I_inj=2):
        dm = (self.alpha_m(self.v) * (1 - self.m) - self.beta_m(self.v) * self.m)*self.dt
        dn = (self.alpha_n(self.v) * (1 - self.n) - self.beta_n(self.v) * self.n)*self.dt
        dh = (self.alpha_h(self.v) * (1 - self.h) - self.beta_h(self.v) * self.h)*self.dt
        
        i = (self.g_Na*(self.m**3)*self.h*(self.v - self.E_Na) + \
             self.g_K*(self.n**4)*(self.v - self.E_K) + \
             self.g_L*(self.v - self.E_L))
             
        I_tot = i - I_inj
                
        dV = -I_tot * self.dt / self.C
        self.m += dm
        self.n += dn
        self.h += dh
        self.v += dV
        return
    
    def sweep(self, func, low, high, step):
        lin_space = np.arange(low, high, step)
        data = func(lin_space)
        return data
    
    def init_equations(self):
        x = sym.symbols('x')
        print("alpha_n",sym.sympify(self.conf["alpha_n"]))
        print("beta_n" ,sym.sympify(self.conf["beta_n" ]))
        print("alpha_m",sym.sympify(self.conf["alpha_m"]))
        print("beta_m" ,sym.sympify(self.conf["beta_m" ]))
        print("alpha_h",sym.sympify(self.conf["alpha_h"]))
        print("beta_h" ,sym.sympify(self.conf["beta_h" ]))
        self.alpha_n = sym.lambdify(x, sym.sympify(self.conf["alpha_n"]),"numpy")
        self.beta_n  = sym.lambdify(x, sym.sympify(self.conf["beta_n" ]),"numpy")
        self.alpha_m = sym.lambdify(x, sym.sympify(self.conf["alpha_m"]),"numpy")
        self.beta_m  = sym.lambdify(x, sym.sympify(self.conf["beta_m" ]),"numpy")
        self.alpha_h = sym.lambdify(x, sym.sympify(self.conf["alpha_h"]),"numpy")
        self.beta_h  = sym.lambdify(x, sym.sympify(self.conf["beta_h" ]),"numpy")
    
    