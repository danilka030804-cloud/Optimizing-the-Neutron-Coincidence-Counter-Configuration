import numpy as np
import subprocess


class set_par:
    def __init__(self):
        pass

    # ---------- ИСТОЧНИК ----------
    def set_pos_src(self, new_value):
        """Источник и палка-толкалка"""
        param_names = [
            'surf 71 cyly 0.0 ', 'surf 72 cyly 0.0 ',
            'surf 73 cyly 0.0 ', 'surf 74 cyly 0.0 ',
            'surf 4 sph 0.0 -0.3 ', 'sp  0.0 0.0 '
        ]
        with open('/home/ndo002/smart_optim/experiment.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            for i, param_name in enumerate(param_names):
                if param_name in line:
                    if i == 0:
                        line = f"{param_name}{new_value} 2.0 -60.8 9.2\n"
                    elif i == 1:
                        line = f"{param_name}{new_value} 1.9 -60.8 8.2\n"
                    elif i == 2:
                        line = f"{param_name}{new_value} 1.85 -60.8 8.2\n"
                    elif i == 3:
                        line = f"{param_name}{new_value} 1.75 -2.3 1.7\n"
                    elif i == 4:
                        line = f"{param_name}{new_value} 0.725\n"
                    elif i == 5:
                        line = f"{param_name}{new_value}\n"
                    break
            new_lines.append(line)

        with open('/home/ndo002/smart_optim/experiment.txt', 'w', encoding='utf-8') as file:
            file.writelines(new_lines)

    # ---------- ТОЛЩИНА КАДМИЯ ----------
    def set_s_Cd(self, new_value):
        """Толщина кадмия"""
        param_name = 'surf 6 cuboid -50.2 50.2 -50.1 50.1 130.7 '
        with open('/home/ndo002/smart_optim/experiment.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            if param_name in line:
                line = f"{param_name}{130.7+new_value}\n"
            new_lines.append(line)

        with open('/home/ndo002/smart_optim/experiment.txt', 'w', encoding='utf-8') as file:
            file.writelines(new_lines)
            

    # ---------- ТОЛЩИНА ПОЛИЭТИЛЕНА ----------
    def set_s_poly(self, new_value):
        """Толщина кадмия"""
        param_name = 'surf 7 cuboid -13.7 14.0 -20.9 22.1 130.8 '
        with open('/home/ndo002/smart_optim/experiment.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            if param_name in line:
                line = f"{param_name}{new_value}\n"
            new_lines.append(line)

        with open('/home/ndo002/smart_optim/experiment.txt', 'w', encoding='utf-8') as file:
            file.writelines(new_lines)

    # ---------- ПРИЗМА ----------
    def set_s_h_prisma(self, s, h):
        """Ширина и высота призмы"""
        s = s / 2
        h = h
        param_name = 'surf 1 cuboid '
        with open('/home/ndo002/smart_optim/experiment.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            if param_name in line:
                line = f"{param_name}{-s} {s} {-s} {s} {h} 130.7\n"
            new_lines.append(line)

        with open('/home/ndo002/smart_optim/experiment.txt', 'w', encoding='utf-8') as file:
            file.writelines(new_lines)

    # ---------- КАМЕРА С ТОПЛИВОМ ----------
    def set_s_h_top(self, a, h):
        """Ширина и высота камеры с топливом"""
        a, b = a / 2, a / 2 + 0.5
        h = h
        param_names = [
            'surf 16 cuboid ', 'surf 51 cuboid ', 'surf 52 cuboid ',
            'surf 53 cuboid ', 'surf 54 cuboid '
        ]
        with open('/home/ndo002/smart_optim/experiment.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            for i, param_name in enumerate(param_names):
                if param_name in line:
                    if i == 0:
                        line = f"{param_name}{-a} {a} {-b} {b} {h} 130.7\n"
                    elif i == 1:
                        line = f"{param_name}{-a} {a} {b-0.7} {b} {h} {h+0.4}\n"
                    elif i == 2:
                        line = f"{param_name}{-a} {a} {-b} {-b+0.7} {h} {h+0.4}\n"
                    elif i == 3:
                        line = f"{param_name}{-a} {-a+0.7} {-b} {b} {h+0.4} {h+0.8}\n"
                    elif i == 4:
                        line = f"{param_name}{a-0.7} {a} {-b} {b} {h+0.4} {h+0.8}\n"
                    break
            new_lines.append(line)

        with open('/home/ndo002/smart_optim/experiment.txt', 'w', encoding='utf-8') as file:
            file.writelines(new_lines)

    # ---------- СЧЕТЧИКИ ----------

    def set_counters(self, n, l, h):  # число и положения счетчиков
        if n > 1:
            s = (27.7 - 3.3 * n) / (n - 1) * l
        else:
            s = 0
        param_names1 = ['surf 8 cuboid ', 'surf 9 cuboid ', 'surf 10 cuboid ', 'surf 11 cuboid ', 'surf 80 cuboid ', 'surf 90 cuboid ', 'surf 100 cuboid ', 'surf 110 cuboid ']
        param_names2 = ['surf 12 cyly ', 'surf 13 cyly ', 'surf 14 cyly ', 'surf 15 cyly ', 'surf 120 cyly ', 'surf 130 cyly ', 'surf 140 cyly ', 'surf 150 cyly ']
        cell_names = ['cell 61 0 void -8 12', 'cell 71 0 He -12', 'cell 62 0 void -9 13', 'cell 72 0 He -13', 'cell 63 0 void -10 14', 'cell 73 0 He -14', 'cell 64 0 void -11 15', 'cell 74 0 He -15', 'cell 65 0 void -80 120', 'cell 75 0 He -120', 'cell 66 0 void -90 130', 'cell 76 0 He -130', 'cell 67 0 void -100 140', 'cell 77 0 He -140', 'cell 68 0 void -110 150', 'cell 78 0 He -150']
        with open('/home/ndo002/smart_optim/experiment.txt', 'r') as file:
            lines = file.readlines()
            new_lines = []
            for line in lines:
                for i, param_name in enumerate(cell_names):
                    if param_name in line:
                        line = f"{param_name}\n"
                        break
                if n%2 == 0:
                    for i, param_name in enumerate(param_names1):
                        if param_name in line:
                            if i == 0:
                                line = f"{param_name}{s/2} {s/2+3.3} -20.9 22.1 {h-1.5} {h+1.5}\n"
                            elif i == 1:
                                line = f"{param_name}{-s/2-3.3} {-s/2} -20.9 22.1 {h-1.5} {h+1.5}\n"
                            break
                    for i, param_name in enumerate(param_names2):
                        if param_name in line:
                            if i == 0:
                                line = f"{param_name}{s/2+1.65} {h} 1.5 -20.9 22.1\n"
                            elif i == 1:
                                line = f"{param_name}{-s/2-1.65} {h} 1.5 -20.9 22.1\n"
                            break
                    if n > 2:
                        for i, param_name in enumerate(param_names1):
                            if param_name in line:
                                if i == 2:
                                    line = f"{param_name}{s/2+(3.3+s)*1} {s/2+3.3+(3.3+s)*1} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                elif i == 3:
                                    line = f"{param_name}{-s/2-3.3-(3.3+s)*1} {-s/2-(3.3+s)*1} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                break
                        for i, param_name in enumerate(param_names2):
                            if param_name in line:
                                if i == 2:
                                    line = f"{param_name}{s/2+1.65+(3.3+s)*1} {h} 1.5 -20.9 22.1\n"
                                elif i == 3:
                                    line = f"{param_name}{-s/2-1.65-(3.3+s)*1} {h} 1.5 -20.9 22.1\n"
                                break
                        if n > 4:
                            for i, param_name in enumerate(param_names1):
                                if param_name in line:
                                    if i == 4:
                                        line = f"{param_name}{s/2+(3.3+s)*2} {s/2+3.3+(3.3+s)*2} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                    elif i == 5:
                                        line = f"{param_name}{-s/2-3.3-(3.3+s)*2} {-s/2-(3.3+s)*2} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                    break
                            for i, param_name in enumerate(param_names2):
                                if param_name in line:
                                    if i == 4:
                                        line = f"{param_name}{s/2+1.65+(3.3+s)*2} {h} 1.5 -20.9 22.1\n"
                                    elif i == 5:
                                        line = f"{param_name}{-s/2-1.65-(3.3+s)*2} {h} 1.5 -20.9 22.1\n"
                                    break
                            if n > 6:
                                for i, param_name in enumerate(param_names1):
                                    if param_name in line:
                                        if i == 6:
                                            line = f"{param_name}{s/2+(3.3+s)*3} {s/2+3.3+(3.3+s)*3} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                        elif i == 7:
                                            line = f"{param_name}{-s/2-3.3-(3.3+s)*3} {-s/2-(3.3+s)*3} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                        break
                                for i, param_name in enumerate(param_names2):
                                    if param_name in line:
                                        if i == 6:
                                            line = f"{param_name}{s/2+1.65+(3.3+s)*3} {h} 1.5 -20.9 22.1\n"
                                        elif i == 7:
                                            line = f"{param_name}{-s/2-1.65-(3.3+s)*3} {h} 1.5 -20.9 22.1\n"
                                        break
                                if 'cell 5 0 pol -7 6 ' in line:
                                    line = f"cell 5 0 pol -7 6 8 9 10 11 80 90 100 110\n"
                                    
                            else:
                                for param_name in cell_names[2*n:]:
                                    if param_name in line:
                                        line = f"%{param_name}\n"
                                        break
                                if 'cell 5 0 pol -7 6 ' in line:
                                    line = f"cell 5 0 pol -7 6 8 9 10 11 80 90% 100 110\n"
                                    
                        else:
                            for param_name in cell_names[2*n:]:
                                if param_name in line:
                                    line = f"%{param_name}\n"
                                    break
                            if 'cell 5 0 pol -7 6 ' in line:
                                line = f"cell 5 0 pol -7 6 8 9 10 11% 80 90 100 110\n"
                                
                    else:
                        for param_name in cell_names[2*n:]:
                            if param_name in line:
                                line = f"%{param_name}\n"
                                break
                        if 'cell 5 0 pol -7 6 ' in line:
                            line = f"cell 5 0 pol -7 6 8 9% 10 11 80 90 100 110\n"
                            
                else:
                    for i, param_name in enumerate(param_names1):
                        if param_name in line:
                            if i == 0:
                                line = f"{param_name}{-1.65} {1.65} -20.9 22.1 {h-1.5} {h+1.5}\n"
                            break
                    for i, param_name in enumerate(param_names2):
                        if param_name in line:
                            if i == 0:
                                line = f"{param_name}{0} {h} 1.5 -20.9 22.1\n"
                            break
                    if n > 1:
                        for i, param_name in enumerate(param_names1):
                            if param_name in line:
                                if i == 1:
                                    line = f"{param_name}{-1.65+(3.3+s)*1} {1.65+(3.3+s)*1} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                elif i == 2:
                                    line = f"{param_name}{-1.65-(3.3+s)*1} {1.65-(3.3+s)*1} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                break
                        for i, param_name in enumerate(param_names2):
                            if param_name in line:
                                if i == 1:
                                    line = f"{param_name}{(3.3+s)*1} {h} 1.5 -20.9 22.1\n"
                                elif i == 2:
                                    line = f"{param_name}{-(3.3+s)*1} {h} 1.5 -20.9 22.1\n"
                                break
                        if n > 3:
                            for i, param_name in enumerate(param_names1):
                                if param_name in line:
                                    if i == 3:
                                        line = f"{param_name}{-1.65+(3.3+s)*2} {1.65+(3.3+s)*2} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                    elif i == 4:
                                        line = f"{param_name}{-1.65-(3.3+s)*1} {1.65-(3.3+s)*1} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                    break
                            for i, param_name in enumerate(param_names2):
                                if param_name in line:
                                    if i == 3:
                                        line = f"{param_name}{(3.3+s)*2} {h} 1.5 -20.9 22.1\n"
                                    elif i == 4:
                                        line = f"{param_name}{-(3.3+s)*2} {h} 1.5 -20.9 22.1\n"
                                    break
                            if n > 5:
                                for i, param_name in enumerate(param_names1):
                                    if param_name in line:
                                        if i == 5:
                                            line = f"{param_name}{-1.65+(3.3+s)*3} {1.65+(3.3+s)*3} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                        elif i == 6:
                                            line = f"{param_name}{-1.65-(3.3+s)*1} {1.65-(3.3+s)*1} -20.9 22.1 {h-1.5} {h+1.5}\n"
                                        break
                                for i, param_name in enumerate(param_names2):
                                    if param_name in line:
                                        if i == 5:
                                            line = f"{param_name}{(3.3+s)*3} {h} 1.5 -20.9 22.1\n"
                                        elif i == 6:
                                            line = f"{param_name}{-(3.3+s)*3} {h} 1.5 -20.9 22.1\n"
                                        break
                                if 'cell 5 0 pol -7 6 ' in line:
                                    line = f"cell 5 0 pol -7 6 8 9 10 11 80 90 100% 110\n"
                                    
                            else:
                                for param_name in cell_names[2*n:]:
                                    if param_name in line:
                                        line = f"%{param_name}\n"
                                        break
                                if 'cell 5 0 pol -7 6 ' in line:
                                    line = f"cell 5 0 pol -7 6 8 9 10 11 80% 90 100 110\n"
                                    
                        else:
                            for param_name in cell_names[2*n:]:
                                if param_name in line:
                                    line = f"%{param_name}\n"
                                    break
                            if 'cell 5 0 pol -7 6 ' in line:
                                line = f"cell 5 0 pol -7 6 8 9 10% 11 80 90 100 110\n"
                                
                    else:
                        for param_name in cell_names[2*n:]:
                                if param_name in line:
                                    line = f"%{param_name}\n"
                                    break
                        if 'cell 5 0 pol -7 6 ' in line:
                            line = f"cell 5 0 pol -7 6 8% 9 10 11 80 90 100 110\n"
                            
                new_lines.append(line)
            with open('/home/ndo002/smart_optim/experiment.txt', 'w', encoding='utf-8') as file:
                file.writelines(new_lines)
    
    # ---------- ГРУППИРОВКА ----------
    def adapt_params(self, optuna_params):
        grouped = {
            'pos_src': optuna_params['pos_src'],
            's_Cd': optuna_params['s_Cd'],
            's_h_prisma': [
                optuna_params['s_h_prisma_s'],
                optuna_params['s_h_prisma_h']
            ],
            'counters': [
                optuna_params['counters_n'],
                optuna_params['counters_l'],
                optuna_params['counters_h']
            ],
            's_h_top': [
                optuna_params['s_h_top_a'],
                optuna_params['s_h_top_h']
            ],
            's_h_poly': optuna_params['s_poly']
        }
        return grouped

    def main(self, params):
        params = self.adapt_params(params)
        for name, value in params.items():
            if name == 'pos_src':
                self.set_pos_src(value)
            elif name == 's_Cd':
                self.set_s_Cd(value)
            elif name == 's_h_prisma':
                self.set_s_h_prisma(value[0], value[1])
            elif name == 's_h_top':
                self.set_s_h_top(value[0], value[1])
            elif name == 'counters':
                self.set_counters(value[0], value[1], value[2])
            elif name == 's_h_poly':
                self.set_s_poly(value)


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

def set_neutrons(n):  # число нейтронов
    param_name = 'set nps '
    with open('/home/ndo002/smart_optim/experiment.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if param_name in line:
            line = f"{param_name}{int(n)} 500\n"
        new_lines.append(line)

    with open('/home/ndo002/smart_optim/experiment.txt', 'w', encoding='utf-8') as file:
        file.writelines(new_lines)



def prin_file():
    """Чтение выходного файла det0.m"""
    with open('/home/ndo002/smart_optim/experiment.txt_det0.m', 'r', encoding='utf-8') as file:
        for line in file:
            if '    1    1    1    1     1     1     1    1    1    1  ' in line:
                parts = line.split()
                number1 = float(parts[-2])
                number2 = float(parts[-1])
                return number1, number2
    raise ValueError("Не найдена строка с данными в det0.m")


def set_on_off(i, h):
    h = (h + 130.7) / 2
    param_name = 'trans U 77 0.00 0.0 '
    with open('/home/ndo002/smart_optim/experiment.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if param_name in line:
            if i:
                line = f"{param_name}{h-3}\n"   # i = 1 — источник включён
            else:
                line = f"{param_name}0.0\n"     # i = 0 — источник выключен
        new_lines.append(line)

    with open('/home/ndo002/smart_optim/experiment.txt', 'w', encoding='utf-8') as file:
        file.writelines(new_lines)



def run_black_box(h):
    """Основной вызов Serpent + обработка результатов"""
    num = np.empty(2)
    delt = np.empty(2)

    for i in range(2):
        set_on_off(i, h)

        subprocess.run(
            ["/home/SHARED/Serpent/sss2_32_fix", "-omp", "35", "/home/ndo002/smart_optim/experiment.txt"],
            check=True,
            stdout=subprocess.DEVNULL
        )

        num[i], delt[i] = prin_file()

    return num[1], num[0], delt[1], delt[0]