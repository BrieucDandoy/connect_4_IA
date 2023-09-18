import numpy as np
import torch
from net import DQN

class connect4:
    def __init__(self) -> None:
        self.grille = np.zeros((6,7))
        self.level = 7*[0]


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai1 = DQN((6, 7),action_size=7).to(device=self.device)
        self.ai1.load_state_dict(torch.load('model/agent1'))
        self.ai1.eval()
        self.ai2 = DQN((6, 7),action_size=7).to(device=self.device)
        self.ai2.load_state_dict(torch.load('model/agent2'))
        self.ai2.eval()
    def place_tile(self,symbol,column_number):
        try:
            column_number = int(column_number)
        except Exception as e:
            print(e)
            print(column_number)
        
        if self.level[column_number] == 6:
            return False
        row_number = self.level[column_number]
        self.grille[row_number][column_number] = symbol
        self.level[column_number] += 1
        return True
 
    def check_win_after_play(self,row_number,column_number,symbol,changed=True):


        check_grid = self.grille.copy()
        if not changed:
            check_grid[row_number][column_number] = symbol

        # Horizontal

        cpt_horizontal = 0
        for idx in range(5):
            if cpt_horizontal == 4:
                return True
            if check_grid[row_number-1][idx] == symbol:
                cpt_horizontal += 1
            else:
                cpt_horizontal = 0
        

        # Vertical

        cpt_vertical = 0
        for idx in range(6):
            if cpt_vertical == 4:
                return True
            if check_grid[idx][column_number] == symbol:
                cpt_vertical += 1
            else:
                cpt_vertical = 0


        # Diagonales

        cpt_diagonale = 0
        for i in range(-7,7):
            if cpt_diagonale == 4:
                return True
            try:
                if check_grid[idx+i][column_number+i] == symbol:
                    cpt_diagonale += 1
                else:
                    cpt_diagonale = 0
            except:
                pass

        cpt_diagonale=0

        for i in range(-7,7):
            if cpt_diagonale == 4:
                return True
            try:
                if check_grid[idx-i][column_number+i] == symbol:
                    cpt_diagonale += 1
                else:
                    cpt_diagonale = 0
            except:
                pass

        return False
    
    def start_game_console(self):
        while True:
            print("\n\n")
            player1_play =  self.play('x',1)
            self.display_grille()
            if player1_play:
                break
            player2_play = self.play('o',2)
            self.display_grille()
            if player2_play:
                break
    
    def display_grille(self):
        print(self.level)
        print('\n')
        print(self.grille)
        print('\n')
    
    def play(self,symbol,id):
        coup_valide = False
        while not coup_valide:
            player_play = input(f"Joueur {id} : jouer un coup, donner un nombre entre 0 et 6 (0 et 6 inclus)")
            player_play = int(player_play)
            if -1 < player_play < 7 and self.level[player_play] < 6:
                coup_valide = True
                print(self.level[player_play])
            else:
                print('coup non-valide\n')

        self.place_tile(symbol=id,column_number=player_play)
        row_number = self.level[player_play]
        win = self.check_win_after_play(column_number=player_play,symbol=symbol,row_number=row_number)
        if win:
            print('Le joueur 1 a gagné')
            return True
        return False


    def get_reward(self,grille_before,col,win):
        if win:
            return 100
        else:
            return 10
        

    def play_NN(self,col,symbol):
        grille_before = self.grille
        valid = self.place_tile(column_number=col,symbol=symbol)
        if valid:
            row_number = self.level[col]
            win = self.check_win_after_play(column_number=col,symbol=symbol,row_number=row_number)
            reward = self.get_reward(grille_before=grille_before,col=col,win=win)
            if win:
                done = True
            else:
                done=False
        else:
            reward = -1000
            done = True
        return self.grille, reward, done
        



    def play_against_ai(self):
        while True:
            print("\n\n")
            player1_play =  self.play(symbol=1,id=1)
            self.display_grille()
            if player1_play:
                break
            state = torch.FloatTensor(self.grille).unsqueeze(0).to(self.device)
            coup = self.ai2(state).argmax()
            self.play_NN(coup,0.5)
            print(f"L'IA a joué le coup {coup}")
            print(self.grille,"\n\n")




    def get_random_position(self,seed=None):
        if seed is not None:
            np.random.seed(seed)
        number_of_tiles = np.random.randint(1,20)

        for _ in range(number_of_tiles):
            not_good_number = True
            idx=0
            while not_good_number and idx < 7:
                idx+=1
                column_player_1 = np.random.randint(0,7)
                if self.level[column_player_1] != 6 and not self.check_win_after_play(column_number=column_player_1,
                                                                                      row_number=self.level[column_player_1],
                                                                                      symbol=0.5,
                                                                                      changed = False):
                    not_good_number = False
            if idx==7:
                return
            row = self.level[column_player_1]
            self.grille[row][column_player_1] = 0.5
            self.level[column_player_1] += 1




            idx = 0
            not_good_number = True
            while not_good_number and idx<7:
                idx+=1
                column_player_2 = np.random.randint(0,7)
                if self.level[column_player_2] < 6 and not self.check_win_after_play(column_number=column_player_2,
                                                                                      row_number=self.level[column_player_2],
                                                                                                            symbol=1,
                                                                                                            changed = False):
                    not_good_number = False
            if idx == 7:
                return
            row = self.level[column_player_2]
            self.grille[row][column_player_2] = 1
            self.level[column_player_2] += 1






if __name__ == '__main__':
    jeu = connect4()
    jeu.get_random_position()
    print(jeu.grille)