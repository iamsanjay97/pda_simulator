## A Naive Congestion Pricing Genco model
class NCpGenco:
    def __init__(self):
        self.seed_value = 0 # seed value will be overidden
        self.total_quantity = 1500 
        self.cleared_quantity = 0
        self.knee_demand = 900 # Congesiton threshold
        self.knee_slope = 5
        self.min_ask_quant = 15
        self.a = 0.00004132
        self.b = 0.00182
        self.c = 14
        self.p = 6
        
    
    def scale_coeff(self):
        y = self.a * self.knee_demand *self.knee_demand + self.b * self.knee_demand + self.c
        self.a *= self.knee_slope
        self.b *= self.knee_slope
        self.c = y - self.a * self.knee_demand * self.knee_demand - self.b * self.knee_demand        
        self.p *= 2

        
    def quad_function(self,cum_quantity):
        a,b,c,p = self.a,self.b,self.c,self.p
        if cum_quantity >= self.knee_demand: 
            self.scale_coeff()
                    
        q = cum_quantity
        price = self.a*(q**2)+self.b*q+self.c
        quantity = max((-self.b/(2.0*self.a) + math.sqrt(self.b**2+4.0*self.a*(self.a*(q**2)+self.b*q+self.p))/(2.0*self.a) - q) , self.min_ask_quant)
        self.a,self.b,self.c,self.p = a,b,c,p
        return price, quantity
    
    # def get_no_of_asks(self):
    #     return len(self.asks)

    def asks(self):
        ask_prices = []
        ask_quantities = []
        cum_quantity = self.cleared_quantity
        while cum_quantity < self.total_quantity:
            price, quantity = self.quad_function(cum_quantity)
            cum_quantity += quantity
            ask_prices.append(price)
            ask_quantities.append(quantity)

        return np.column_stack((ask_prices,ask_quantities))