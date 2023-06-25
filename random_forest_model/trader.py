class Trader():
    def __init__(self, config, model):
        self.config = config | model.config
        self.model = model

    def trade(self):
        self.sell_trades = self.model.sb_FL[0].getTrades(self.model.sell_probs, self.config["N_MAX_PORTFOLIO"], self.config["CONF"], allow_early_unwind=False)
        self.buy_trades = self.model.sb_FL[1].getTrades(self.model.buy_probs, self.config["N_MAX_PORTFOLIO"], self.config["CONF"], allow_early_unwind=False)

        return self.sell_trades, self.buy_trades

