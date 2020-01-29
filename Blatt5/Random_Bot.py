class my_random_bot:
    def __init__(self, spieler_farbe):
        """
        :param spieler_farbe: Entweder "black" oder "white"

        Die Klasse my_random_bot benötigt außerdem die Attribute:
            self.pos_felder:    Hält stets die möglichen Felder, zu Beginn None
            self.cur_choice:    Die aktuell beste Wahl, zu Beginn None
            self.timeout:       Wird in der Klasse Spielfeld nach ablauf des Timeouts gesetzt. Falls also Schleifen
                                verwendet werden, prüfen Sie bitte stets auch die Timeout Bedingung, zu Beginn False
        """
        pass

    def set_next_stone(self):
        """
        Setzen Sie den Parameter self.cur_choice. Ein Tuple von der Form (idy, idx) wobei idy der Zeile und idx der Spalte entspricht.
        """
        pass


