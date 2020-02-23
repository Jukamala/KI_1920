import time
import numpy as np
import pygame
from Bot_42 import my_Bot
from TwoBot import my_Bot as my_Bot_2
from Greedy_Bot import my_Bot as my_Bot_3

from pygame import gfxdraw
import threading


class Spielfeld:
    def __init__(self):
        self.spielfeld = self.new_Spielfeld().copy()

        self.background_color = (49, 150, 100)
        self.line_color = (0, 0, 0)

        pygame.init()
        self.display_height = 640
        self.display_width = 640
        self.zellen_height = int(float(self.display_height) / 8.0)
        self.zellen_width = int(float(self.display_width) / 8.0)
        self.stein_radius = 35

        self.game_display = pygame.display.set_mode(size=(self.display_width, self.display_height))
        pygame.display.set_caption("Othello")
        self.clock = pygame.time.Clock()
        self.hat_gewonnen = False

    def new_Spielfeld(self):

        new_feld = np.array(["empty"] * (8 * 8)).reshape((8, 8))

        new_feld[3][3] = "white"
        new_feld[4][4] = "white"
        new_feld[3][4] = "black"
        new_feld[4][3] = "black"
        """
        new_feld = np.array(["black" if np.random.uniform(0,1) < 1.5 else "white" for elem in range(8*8)]).reshape((8, 8))
        new_feld[7][7] = "empty"
        # new_feld[7][7] = "black"
        # new_feld[6][6] = "white"
        # new_feld[7][6] = "white"
        # new_feld[6][7] = "white"
        """
        # print(new_feld)
        # self.spielfeld = new_feld.copy()
        return new_feld

    def update_spielfeld(self, black_stones, white_stones, spieler, pos_felder):
        if spieler == 0:
            pygame.display.set_caption("Spieler 1 (schwarz) ist dran.")
        else:
            pygame.display.set_caption("Spieler 2 (weiß) ist dran.")
        self.game_display.fill(self.background_color)
        # x_coordiantes = [elem for elem in np.arange(0,self.display_width, self.display_width / 8)]
        # y_coordiantes = [elem for elem in np.arange(0, self.display_height, self.display_height / 8)]
        # coordiantes = [(x,y) for x in x_coordiantes for y in y_coordiantes]
        # (x,y)
        coordiantes = [((0, 0),(0,640)), ((80, 0),(80,640)), ((160, 0),(160,640)), ((240, 0),(240,640)), ((320, 0),(320,640)), ((400, 0),(400,640)), ((480, 0),(480,640)), ((560, 0),(560,640)), ((638, 0),(638,638)),
                       ((0, 0),(640,0)), ((0, 80),(640,80)), ((0, 160),(640,160)), ((0, 240),(640,240)), ((0, 320),(640,320)), ((0, 400),(640,400)), ((0, 480),(640,480)), ((0, 560),(640,560)), ((0, 638),(638,638))]
        # print("coordiantes =", coordiantes)

        for elem in coordiantes:
            pygame.draw.line(self.game_display, self.line_color, elem[0], elem[1], 2)

        for idy, row in enumerate(self.spielfeld):
            for idx, zelle in enumerate(row):
                # print("zelle =", zelle)
                if zelle == "white":
                    offset_width = int((self.zellen_width - 2*self.stein_radius)/2.0)
                    offset_height = int((self.zellen_height - 2 * self.stein_radius) / 2.0)
                    pos = (int(float(idx+1)/8.0 * self.display_width) - (self.stein_radius+offset_width),int(float(idy+1)/8.0 * self.display_height)-(self.stein_radius+offset_height))
                    # pygame.draw.circle(self.game_display, (255,255,255), pos, self.stein_radius)
                    pygame.gfxdraw.aacircle(self.game_display, pos[0], pos[1], self.stein_radius, (255,255,255))
                    pygame.gfxdraw.filled_circle(self.game_display,  pos[0], pos[1], self.stein_radius, (255, 255, 255))
                elif zelle == "black":
                    offset_width = int((self.zellen_width - 2*self.stein_radius)/2.0)
                    offset_height = int((self.zellen_height - 2 * self.stein_radius) / 2.0)
                    pos = (int(float(idx+1)/8.0 * self.display_width) - (self.stein_radius+offset_width),int(float(idy+1)/8.0 * self.display_height)-(self.stein_radius+offset_height))
                    # pygame.draw.circle(self.game_display, (0,0,0), pos, self.stein_radius)
                    pygame.gfxdraw.aacircle(self.game_display, pos[0], pos[1], self.stein_radius, (0, 0, 0))
                    pygame.gfxdraw.filled_circle(self.game_display, pos[0], pos[1], self.stein_radius, (0, 0, 0))

        for idy, row in enumerate(self.spielfeld):
            for idx, zelle in enumerate(row):
                if pos_felder[idy][idx] == "pos_f":
                    offset_width = int((self.zellen_width - 2 * self.stein_radius) / 2.0)
                    offset_height = int((self.zellen_height - 2 * self.stein_radius) / 2.0)
                    pos = (int(float(idx + 1) / 8.0 * self.display_width) - (self.stein_radius + offset_width),
                           int(float(idy + 1) / 8.0 * self.display_height) - (self.stein_radius + offset_height))
                    # pygame.draw.circle(self.game_display, (50, 50, 50), pos, 5)
                    pygame.gfxdraw.aacircle(self.game_display, pos[0], pos[1], 5, (50, 50, 50))
                    pygame.gfxdraw.filled_circle(self.game_display, pos[0], pos[1], 5, (50, 50, 50))

        if self.hat_gewonnen:

            if black_stones > white_stones:
                gewinner_text = "Spieler 1 (schwarz) hat gewonnen."
                line2_pos = (85, 250)
            elif black_stones < white_stones:
                gewinner_text = "Spieler 2 (weiß) hat gewonnen."
                line2_pos = (100, 250)
            else:
                gewinner_text = "das Spiel endet unentschieden."
                line2_pos = (110, 260)

            delta_h = 100
            delta_w = 150
            menu = pygame.Surface((640,640))
            menu.set_alpha(180)
            menu.fill((0, 0, 0))
            self.game_display.blit(menu, (0, 0))
            # col = pygame.Color(128, 128, 128)
            # pygame.Color
            pygame.draw.polygon(menu, [128,128,128], [(delta_w,delta_h),(640-delta_w,delta_h),(640-delta_w,640-delta_h),(delta_w,640-delta_h)], 0)

            pygame.font.init()
            myfont = pygame.font.SysFont('Comic Sans MS', 40)
            textsurface_line1 = myfont.render("Herzlichen Glückwunsch", False, (200, 200, 200))
            myfont = pygame.font.SysFont('Comic Sans MS', 30)
            textsurface_line2 = myfont.render(gewinner_text, False, (200, 200, 200))
            textsurface_line3 = myfont.render("#schwarze Steine " + str(black_stones) , False, (200, 200, 200))
            textsurface_line4 = myfont.render("#weiße Steine " + str(white_stones), False, (200, 200, 200))
            self.game_display.blit(textsurface_line1, (95, 145))
            self.game_display.blit(textsurface_line2, line2_pos)
            self.game_display.blit(textsurface_line3, (180, 350))
            self.game_display.blit(textsurface_line4, (199, 400))

        # pygame.display.flip()
        pygame.display.update()

    def coordinate_to_ids(self, tuple):
        x, y = tuple
        idx = int(float(x)/float(self.display_width) * 8)
        idy = int(float(y) / float(self.display_height) * 8)
        # print("idx =", idx)
        # print("idy =", idy)
        return idx, idy

    def check_rules(self, neuer_stein, spieler, check_only=False):
        """
        :param neuer_stein: Tupel von der Form (x,y) mit x und y im Intervall [0,7]. Wobei x der Spalte
                            und y der Zeile entspricht. Beispiel: self.spielfeld[y][x]
                            Zum aktualisieren des Spielfeldes genügt es, wenn sie das Array self.spielfeld aktualisieren, das
                            heißt, Sie müssen nicht die Methode update_spielfeld aufrufen.
        :param spieler: Entspricht dem aktuellen Spieler 0 steht für "black" 1 für "white"
        :param check_only: optinaler Parameter.
                            Wenn der Zug zulässig ist und check_only=False, dann soll self.spielfeld
                            aktualisiert werden. check_only=True, dann wird self.spielfeld nicht aktualisiert
        :return: True, wenn neuer_stein für spieler ein zulässiger Zug ist, Flase sonst
        """
        idx, idy = neuer_stein
        akt_farbe = "black" if spieler == 0 else "white"
        gegner_farbe = "black" if ((spieler + 1) % 2) == 0 else "white"

        # print("\nidy, idx =", idy, idx)
        # print("self.spielfeld[idy][idx] =", self.spielfeld[idy][idx])

        if not self.spielfeld[idy][idx] == "empty":
            return False
        else:
            possible_directions = []  # von (-1,-1) bis (1,1)
            for row in np.arange(max(idy - 1, 0), min(idy + 2, 8), 1):
                for zelle in np.arange(max(idx - 1, 0), min(idx + 2, 8), 1):
                    if not (row == idy and zelle == idx) and self.spielfeld[row][zelle] == gegner_farbe:
                        # print("(row-idy, zelle-idx) =", (row-idy, zelle-idx))
                        possible_directions.append((row - idy, zelle - idx))
            if len(possible_directions) > 0:
                richtiger_zug = False
                for elem in possible_directions:
                    schluss_stein_gefunden = False
                    temp_spielfeld = self.spielfeld.copy()
                    y_direction = idy + elem[0]
                    x_direction = idx + elem[1]
                    while y_direction in range(8) and x_direction in range(8) and not schluss_stein_gefunden:
                        if self.spielfeld[y_direction][x_direction] == gegner_farbe:
                            temp_spielfeld[y_direction][x_direction] = akt_farbe
                        elif self.spielfeld[y_direction][x_direction] == akt_farbe:
                            schluss_stein_gefunden = True
                            richtiger_zug = True
                            if not check_only:
                                self.spielfeld = temp_spielfeld.copy()
                        elif self.spielfeld[y_direction][x_direction] == "empty":
                            break

                        y_direction += elem[0]
                        x_direction += elem[1]
                if richtiger_zug:
                    return True
            return False

    def possible_felder(self, spieler):
        # akt_spielfeld = self.spielfeld.copy()
        pos_felder = np.array(["empty"] * (8 * 8)).reshape((8, 8))
        counter = 0
        for row in range(8):
            for zelle in range(8):
                if self.spielfeld[row][zelle] == "empty" and self.check_rules((zelle, row), spieler, check_only=True):
                    counter += 1
                    pos_felder[row][zelle] = "pos_f"
        # self.spielfeld = akt_spielfeld.copy()
        # print("pos_felder =", counter == len(pos_felder))
        return counter, pos_felder

    def check_for_win(self):
        black_stones = (self.spielfeld == "black").sum()
        white_stones = (self.spielfeld == "white").sum()

        if not("empty" in self.spielfeld) or black_stones == 0 or white_stones == 0:
            return black_stones, white_stones, True
        return black_stones, white_stones, False

    def get_winner(self):
        black_stones, white_stones, _ = self.check_for_win()
        gewinner = "unentschieden"
        if black_stones > white_stones:
            gewinner = "black"
        elif black_stones < white_stones:
            gewinner = "white"
        return gewinner

    def play_game(self, use_bots):
        spieler = 0
        running = True
        timeout = 2

        black_stones, white_stones, self.hat_gewonnen = self.check_for_win()
        akt_spieler_musste_passen = 0

        self.update_spielfeld(black_stones, white_stones, spieler, self.possible_felder(spieler)[1])
        self.clock.tick(30)

        if use_bots:
            """
            Hier können Sie die Bots festlegen, die spielen sollen. Für den Fall, dass Sie gegen
            einen Bot spielen möchten und den Schwierigkeitsgrad des Bots ändern möchten, können
            Sie den bot_white_1 anpassen.
            """
            bot_black_0 = my_Bot(spieler_farbe="black")
            # bot_black_0 = my_greedy_bot(spieler_farbe="black", simple_Bot=True)

            bot_white_1 = my_Bot(spieler_farbe="white")
            # bot_white_1 = my_random_bot(spieler_farbe="white")
            # bot_white_1 = my_greedy_bot(spieler_farbe="white", simple_Bot=False)
            # bot_white_1 = my_minmax_bot(spieler_farbe="white")
            # bot_white_1 = my_learnable_bot(spieler_farbe="white")

        gewinner = ""
        counter = 0
        while running: # and counter < 4:
            counter += 1
            if self.hat_gewonnen and use_bots == 1:
                time.sleep(3)
                running = False
            if self.possible_felder(spieler)[0] == 0 and not self.hat_gewonnen:
                akt_spieler_musste_passen += 1
                spieler = (spieler + 1) % 2
                self.update_spielfeld(black_stones, white_stones, spieler, self.possible_felder(spieler)[1])
                self.clock.tick(30)
            if akt_spieler_musste_passen >= 2:
                self.hat_gewonnen = True
                gewinner = self.get_winner()
            if use_bots == 1 and not self.hat_gewonnen:
                if self.possible_felder(spieler)[0] > 0:
                    if spieler == 0:
                        bot_black_0.spielfeld = self.spielfeld.copy()
                        bot_black_0.pos_felder = self.possible_felder(spieler)[1].copy()
                        t1 = threading.Thread(target=bot_black_0.set_next_stone)
                        t1.start()
                        t1.join(timeout=timeout)

                        try:
                            idy, idx = bot_black_0.cur_choice
                        except:
                            raise Exception("Das Attribut self.cur_choice Ihres Bots wurde nicht rechzeitig gesetzt. Beachten Sie außerdem, dass self.cur_choice ein Tuple der Form (idy, idx) ist.")
                        bot_black_0.timeout = True
                        t1.join()
                    else:
                        bot_white_1.spielfeld = self.spielfeld.copy()
                        bot_white_1.pos_felder = self.possible_felder(spieler)[1].copy()
                        t2 = threading.Thread(target=bot_white_1.set_next_stone)
                        t2.start()
                        t2.join(timeout=timeout)

                        try:
                            idy, idx = bot_white_1.cur_choice
                        except:
                            raise Exception("Das Attribut self.cur_choice Ihres Bots wurde nicht rechzeitig gesetzt. Beachten Sie außerdem, dass self.cur_choice ein Tuple der Form (idy, idx) ist.")
                        bot_white_1.timeout = True
                        t2.join()

                    if self.check_rules((idx, idy), spieler):
                        akt_spieler_musste_passen = 0
                        self.spielfeld[idy][idx] = "black" if spieler == 0 else "white"
                        black_stones, white_stones, self.hat_gewonnen = self.check_for_win()
                        if not self.hat_gewonnen:
                            spieler = (spieler + 1) % 2
                        else:
                            gewinner = self.get_winner()
                    offset_width = int((self.zellen_width - 2 * self.stein_radius) / 2.0)
                    offset_height = int((self.zellen_height - 2 * self.stein_radius) / 2.0)
                    pos = (int(float(idx + 1) / 8.0 * self.display_width) - (self.stein_radius + offset_width),
                           int(float(idy + 1) / 8.0 * self.display_height) - (self.stein_radius + offset_height))
                    pygame.gfxdraw.aacircle(self.game_display, pos[0], pos[1], 15, (150, 50, 50))
                    pygame.gfxdraw.filled_circle(self.game_display, pos[0], pos[1], 15, (150, 50, 50))
                    pygame.display.update()
                    # time.sleep(2)
                    # time.sleep(0.75)

            if use_bots == 2 and not self.hat_gewonnen:
                if self.possible_felder(spieler)[0] > 0:
                    if spieler == 0:
                        spieler_0_ist_dran = True
                        while spieler_0_ist_dran:
                            for event in pygame.event.get():
                                # print(event)
                                if event.type == pygame.QUIT:
                                    running = False
                                    pygame.quit()
                                else:
                                    if event.type == pygame.MOUSEBUTTONDOWN and akt_spieler_musste_passen < 2:
                                        if event.button == 1:
                                            idx, idy = self.coordinate_to_ids(event.pos)
                                            spieler_0_ist_dran = False
                    else:
                        bot_white_1.spielfeld = self.spielfeld.copy()
                        bot_white_1.pos_felder = self.possible_felder(spieler)[1].copy()
                        t1 = threading.Thread(target=bot_white_1.set_next_stone)
                        t1.start()
                        t1.join(timeout=timeout)

                        try:
                            idy, idx = bot_white_1.cur_choice
                        except:
                            raise Exception("Das Attribut self.cur_choice Ihres Bots wurde nicht rechzeitig gesetzt. Beachten Sie außerdem, dass self.cur_choice ein Tuple der Form (idy, idx) ist.")
                        bot_white_1.timeout = True
                        t1.join()
                    if isinstance(idy, tuple):
                        temporary = idy
                        idy = temporary[0]
                        idx = temporary[1]
                    if self.check_rules((idx, idy), spieler):
                        print("[2] idy, idx =", idy, idx)
                        akt_spieler_musste_passen = 0
                        self.spielfeld[idy][idx] = "black" if spieler == 0 else "white"
                        black_stones, white_stones, self.hat_gewonnen = self.check_for_win()
                        if not self.hat_gewonnen:
                            spieler = (spieler + 1) % 2
                        else:
                            gewinner = self.get_winner()
                    offset_width = int((self.zellen_width - 2 * self.stein_radius) / 2.0)
                    offset_height = int((self.zellen_height - 2 * self.stein_radius) / 2.0)
                    pos = (int(float(idx + 1) / 8.0 * self.display_width) - (self.stein_radius + offset_width),
                           int(float(idy + 1) / 8.0 * self.display_height) - (self.stein_radius + offset_height))
                    pygame.draw.circle(self.game_display, (150, 50, 50), pos, 15)
                    pygame.display.update()
                    # time.sleep(1)


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    if event.type == pygame.MOUSEBUTTONDOWN and use_bots == 0 and akt_spieler_musste_passen < 2:
                        if event.button == 1:
                            print("event.pos =", event.pos)
                            idx, idy = self.coordinate_to_ids(event.pos)
                            if self.check_rules((idx, idy), spieler):# and self.possible_felder(spieler)[0] > 0:
                                akt_spieler_musste_passen = 0
                                self.spielfeld[idy][idx] = "black" if spieler == 0 else "white"

                                black_stones, white_stones, self.hat_gewonnen = self.check_for_win()
                                if not self.hat_gewonnen:
                                    spieler = (spieler + 1) % 2
                                else:
                                    gewinner = self.get_winner()

            self.update_spielfeld(black_stones, white_stones, spieler, self.possible_felder(spieler)[1])
            self.clock.tick(30)

        return gewinner


if __name__ == '__main__':
    play_many_games = True
    if play_many_games:
        black_counter = 0
        white_counter = 0
        max_games = 20
        for elem in range(max_games):
            othello = Spielfeld()
            gewinner = othello.play_game(use_bots=1)
            if gewinner == "black":
                black_counter += 1
            elif gewinner == "white":
                white_counter += 1
            print(str(elem + 1) + ". Spiel: Der Gewinner ist", gewinner)
            delay = 0
            for i in range(delay):
                print("\tDas nächste Spiel beginnt in:", delay-i, "Sekunde(n).")
                time.sleep(1)
        print("\n---------------------------------------------")
        print("Schwarz hat", black_counter, "mal gewonnen.")
        print("Weiß hat", white_counter, "mal gewonnen")
        print("Unentschieden", max_games - (white_counter + black_counter), "mal.")
        print("---------------------------------------------")

    else:
        use_bots = 2  # 0 -> two human player, 1 -> two computer player, 2 -> one human and one computer player
        othello = Spielfeld()
        gewinner = othello.play_game(use_bots=use_bots)
        print("Der Gewinner ist:", gewinner)