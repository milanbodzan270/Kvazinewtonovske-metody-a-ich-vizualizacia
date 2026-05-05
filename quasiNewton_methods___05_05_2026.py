#Importovanie potrebných knižníc:
import numpy as np 
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



#Definovanie možných druhov premenných:
n = 100
variables = sp.symbols(f"x1:{n+1}")
x_sym = sp.Matrix(variables)
i = sp.symbols('i', integer=True)
x1, x2 = variables[:2]



#Numerický a symbolický tvar prvej testovacej funkcie:
def f1_num(x):
    return (x[0] - 2)**4 + (x[0] - 2)**2*x[1]**2 + (x[1] + 1)**2    
f1_sym = (x1 - 2)**4 + (x1 - 2)**2*x2**2 + (x2 + 1)**2



#Numerický a symbolický tvar testovacej funkcie Rosenbrock:
def rosenbrock_num(x):
    a = 1
    b = 2
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2
rosenbrock_sym = (1 - x1)**2 + 2*(x2 - x1**2)**2



#Numerický a symbolický tvar testovacej funkcie Himmelblau:
def himmelblau_num(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 11)**2
himmelblau_sym = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 11)**2



#Numerický a symbolický tvar testovacej funkcie Zakharov:
def zakhar_num(x):
    return x[0]**2 + x[1]**2 + (0.5*x[0] + x[1])**2 + (0.5*x[0] + x[1])**4
zakhar_sym = x1**2 + x2**2 + (0.5*x1 + x2)**2 + (0.5*x1 + x2)**4
 


#Počiatočné body:
x0_1 = [3, -2]
x0_2 = [0, 0]
x0_3 = [-0.5, -1.5]
x0_5 = [-2, 3]
x0_6 = [-4, -4]



#Výpočet skutočného riešenia:
    
#Vstupnými parametrami sú - numerický tvar účelovej funkcie
#                         - počiatočný bod x0
def real_result(f_num, x0):
    result = minimize(f_num, x0, method="Nelder-Mead", tol=1e-16)
    x_opt = result.x
    return x_opt



#Skutočné riešenia vybraných testovacích funkcií:
result_f1 = real_result(f1_num, x0_1)
result_rosenbrock = real_result(rosenbrock_num, x0_1)
result_himmelblau_1 = real_result(himmelblau_num, x0_1)
result_himmelblau_2 = real_result(himmelblau_num, x0_5)
result_himmelblau_3 = real_result(himmelblau_num, x0_2)
result_himmelblau_4 = real_result(himmelblau_num, x0_6)
#Body minima funkcie Himmelblau spojíme do vektora.
result_himmelblau = np.array([result_himmelblau_1, result_himmelblau_2, result_himmelblau_3, result_himmelblau_4])
result_zakhar = real_result(zakhar_num, x0_1)



#Výpočet symbolického tvaru gradientu:

#Vstupnými parametrami sú: - symbolický tvar funkcie
#                          - počet premenných
def gradient_sym(f_sym, n):
    
    #Rozlíšenie medzi funkciou zadanou pomocou matice a inou: 
    if isinstance(f_sym, sp.Matrix):
        f_expr = f_sym[0]
    else:
        f_expr = f_sym

    #Prvé parciálne derivácie:        
    g_sym = [sp.diff(f_expr, variable) for variable in variables[:n]]
    return g_sym



#Numerická evaluácia symbolického tvaru gradientu:
    
#Vstupnými parametrami sú: - symbolický tvar gradientu
#                          - počet premenných
def gradient_num(g_sym, n):
    g_num = sp.lambdify(variables[:n], g_sym, "numpy")
    return g_num

      

#Výpočet symbolického tvaru Hesseho matice:
    
#Vstupnými parametrami sú: - symbolický tvar funkcie
#                          - počet premenných
def Hessian_sym(f_sym, n):
    
    #Rozlíšenie medzi funkciou zadanou pomocou matice a inou: 
    if isinstance(f_sym, sp.Matrix):
        f_expr = f_sym[0]
    else:
        f_expr = f_sym
    
    #Druhé parciálne derivácie: 
    H_sym = [[sp.diff(f_expr, variable_1, variable_2) for variable_2 in variables[:n]] for variable_1 in variables[:n]]
    return H_sym



#Numerická evaluácia symbolického tvaru Hesseho matice:
    
#Vstupnými parametrami sú: - symbolický tvar Hesseho matice
#                          - počet premenných
def Hessian_num(H_sym, n):
    H_num = sp.lambdify(variables[:n], H_sym, "numpy")
    return H_num



#Newtonova metóda:
    
#Vstupnými parametrami sú: - numerický tvar účelovej funkcie
#                          - symbolický tvar účelovej funkcie
#                          - počiatočný bod x0
#                          - tolerancia
def Newton_method(f_num, f_sym, x0, tol):
    x = x0.copy()
    n = len(x)
    k = 0
    
    #Výpočet symbolického tvaru gradientu účelovej funkcie:
    g_sym = gradient_sym(f_sym, n)
    
    #Výpočet symbolického tvaru Hesseho matice:
    H_sym = Hessian_sym(f_sym, n)
    
    #Numerická evaluácia symbolického tvaru gradientu:
    g_num = gradient_num(g_sym, n)
    g = np.array(g_num(*x), dtype=float)
    
    x_k = []
    x_k.append(x.copy())
    
    #Zastavovacie kritérium:
    while np.linalg.norm(g) > tol:

        #Numerická evaluácia symbolického tvaru Hesseho matice:
        H_num = Hessian_num(H_sym, n)
        H = np.array(H_num(*x), dtype=float)
        
        #Výpočet nového newtonovského smeru:
        s = -np.linalg.solve(H, g)
        
        #Výpočet nového bodu:
        x += s
        k += 1
        x_k.append(x.copy())
        
        #Numerická evaluácia symbolického tvaru gradientu v novom bode:
        g = np.array(g_num(*x), dtype=float)
        
    #Vypísanie metódy, potrebného počtu iterácií, prípadne nájdeného riešenia:
    print("Newton")
    print(f"Number of iterations: {k}")
    print(f"Optimal solution: {x_k[k]}")
    
    #Výstupom je zoznam bodov z jednotlivých iterácií.
    return x_k

    

#Kvázinewtonovské metódy:
#(Možné voľby metódy: DFP, BFGS, SR1, Hoshino, Broyden (pri tejto metóde je potrebné parameter phi zmeniť v kóde))
#(Je možné zmeniť aj maximálny počet iterácií a hodnotu eps, ktorá slúži na numerické uvoľnenie podmienok)
    
#Vstupnými parametrami sú: - numerický tvar účelovej funkcie
#                          - symbolický tvar účelovej funkcie
#                          - počiatočný bod x0
#                          - tolerancia
#                          - metóda (zadáva sa názov v úvodzovkách, napr. "BFGS" )
def quasiNewton_method(f_num, f_sym, x0, tol, method):
    x = x0.copy()
    n = len(x)
    k = 0
    
    #Tu možno meniť hodnoty pre maximálny počet iterácií a eps.
    k_max = 100
    eps = 1e-15
    
    #Výpočet symbolického tvaru gradientu účelovej funkcie:
    g_sym = gradient_sym(f_sym, n)
    
    #Numerická evaluácia symbolického tvaru gradientu účelovej funckie: 
    g_num = gradient_num(g_sym, n)
    g = np.array(g_num(*x), dtype=float)
    
    #Nastavenie počiatočnej aproximácie Hesseho matice:
    H = np.eye(n)
    
    x_k = []
    x_k.append(x.copy())
    
    #Zastavovacie kritérium:
    while np.linalg.norm(g) > tol:
        
        #Výpočet nového spádového smeru:
        s = - H @ g
        
        #Backtracking (hľadanie približne optimálnej dĺžky kroku)
        #(Možno rozumne meniť hodnoty lambda_k, alpha, delta)
        lambda_k = 1.0
        alpha = 0.25
        delta = 0.8
        while f_num(x + lambda_k * s) > f_num(x) + alpha * lambda_k * (g @ s):
            lambda_k *= delta
        
        #Výpočet nového bodu v ďalšej iterácii:
        k += 1
        x_new = x + lambda_k * s
        
        #Numerická evaluácia symbolického tvaru gradientu účelovej funkcie v novom bode:
        g_new = np.array(g_num(*x_new), dtype=float)
        
        #Spočítanie vektorov y a p:
        y = g_new - g
        p = x_new - x
        
        #Spocítanie korekčnej matice vzhľadom na výber metódy:
        #(Sú tu implementované rôzne poistky proti zlyhaniu.)
        try:
            if method == "DFP":
                pTy = p @ y
                yTHy = y @ H @ y
                if pTy <= eps or np.abs(yTHy) <= eps:
                    print(f"Method failed for x_0 = {x0}. Encountered division by 0.")
                    break
                else:
                    DeltaH = np.outer(p, p) / pTy - np.outer((H @ y), (H @ y)) / yTHy
            elif method == "BFGS":
                pTy = p @ y
                if pTy <= eps:
                    print(f"Method failed for x_0 = {x0}. Encountered division by 0.")
                    break
                else:
                    DeltaH = (1 + (y @ H @ y) / pTy) * (np.outer(p, p) / pTy) - (np.outer(H @ y, p) + np.outer(p, y @ H)) / pTy               
            elif method == "SR1":
                rTy = (p - H @ y) @ y
                if np.abs(rTy) <= eps:
                    print(f"Method failed for x_0 = {x0}. Encountered division by 0.")
                    break
                else:
                    DeltaH = np.outer((p - H @ y), (p - H @ y)) / rTy
            elif method == "Hoshino":
                pTy = p @ y
                rTy = (p + H @ y) @ y
                if pTy <= eps or np.abs(rTy) <= eps:
                    print(f"Method failed for x_0 = {x0}. Encountered division by 0.")
                    break
                else:
                    DeltaH = np.outer(2 * p, p) / pTy - np.outer((p + H @ y), (p + H @ y)) / rTy
            elif method == "Broyden":
                #Tu možno zmeniť hodnotu parametra phi:
                phi = 1
                pTy = p @ y
                yTHy = y @ H @ y
                if pTy <= eps or np.abs(yTHy) <= eps:
                    print(f"Method failed for x_0 = {x0}. Encountered division by 0.")
                    break
                else:
                    DeltaH_DFP = np.outer(p, p) / pTy - np.outer((H @ y), (H @ y)) / yTHy
                    DeltaH_BFGS = (1 + (y @ H @ y) / pTy) * (np.outer(p, p) / pTy) - (np.outer(H @ y, p) + np.outer(p, y @ H)) / pTy
                    DeltaH = (1 - phi)*DeltaH_DFP + phi*DeltaH_BFGS
        except RuntimeWarning:
            break
        
        #Výpočet novej aproximácie Hesseho matice a priradenie nového bodu a príslušného gradientu do ďalšej iterácie:
        H += DeltaH
        g = g_new
        x = x_new
        x_k.append(x.copy())
        
        #Poistka proti príliš veľkému počtu iterácií:
        if k == k_max:
            print(f"Method failed for x_0 = {x0}. Maximum number of iterations was hit.")
            break
    
    #Vypísanie metódy, potrebného počtu iterácií, prípadne nájdeného riešenia: 
    print(f"{method}")
    print(f"Number of iterations: {k}")
    if np.linalg.norm(g) <= tol:
        print(f"Optimal solution: {x}")  
    
    #Výstupom je zoznam bodov z jednotlivých iterácií.
    return x_k



#Vypočítanie chyby optimálneho riešenia v jednotlivých iteráciach:
    
#Vstupnými parametrami sú: - zoznam bodov získaných v jednotlivých iteráciach
#                          - skutočné riešenie (skutočný bod minima)
def distances(x_k, x_opt):
    distances = np.zeros(len(x_k))
    
    #Každý prvok vektora distances je norma rozdielu bodu z konkrétnej iterácie a bodu skutočného minima.
    for k in range(len(x_k)):
        distances[k] = np.linalg.norm(x_k[k] - x_opt)
    return distances



#Vizualizácia vývoja chyby riešenia a konvergenčnej cesty pre metódy DFP, BFGS, SR1, Hoshino a Newtonovu metódu:

#Vstupnými parametrami sú: - numerický tvar účelovej funkcie
#                          - symbolický tvar účelovej funkcie
#                          - počiatočný bod x0
#                          - skutočné riešenie (skutočný bod minima)
#                          - tolerancia
#                          - názov účelovej funckie (zadáva sa v tvare: title = "Názov funckie" )
def visualize(f_num, f_sym, x0, result, tol, title, methods=["DFP", "BFGS", "SR1", "Hoshino", "Newton"]):
    
    #Vytvorenie zoznamu nájdených riešení jednotlivých metód:
    result_methods = {}  
    for method in methods:
        if method == "Newton":
            result_methods[method] = Newton_method(f_num, f_sym, x0, tol)
        else:    
            result_methods[method] = quasiNewton_method(f_num, f_sym, x0, tol, method)
    
    #Vytvorenie zoznamov vývoja chýb riešení naprieč jednotlivými iteráciami a jeho vykreslenie:
    plt.figure(figsize=(9, 5))
    distances_lists = []
    for method in methods:
        distances_method = distances(result_methods[method], result)
        distances_lists.append(distances_method)
        plt.semilogy(range(0, len(distances_method)), distances_method, marker="o", linestyle="dashed", label=method)
    
    plt.xlabel(r"Počet iterácií $k$")
    plt.ylabel(r"$\| x_{k} - x_{\text{opt}} \|$")
    plt.title(f"Vývoj chyby riešenia naprieč jednotlivými iteráciami pre funkciu {title}")
    plt.grid()
    longest_list = max(distances_lists, key=len)
    max_k = len(longest_list)
    plt.xticks(range(max_k + 1))
    plt.legend(fontsize=12, framealpha=1)
    plt.tight_layout()
    plt.show()
    
    #Určenie farebnej škály pre vizuálnu konzistenciu:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    #Vizualizácia konvergenčnej cesty na vrstevnicovej mape:
    for m, method in enumerate(methods):
        result_method = result_methods[method]
        
        #Podmienka, že účelová funckia je dvojrozmerná:
        if len(x0) == 2:
            
            #Vycentrovanie, aby bola viditeľná celá cesta:
            x1_range = (min(np.array(result_method)[:, 0]) - 0.25, max(np.array(result_method)[:, 0]) + 0.25)
            x2_range = (min(np.array(result_method)[:, 1]) - 0.25, max(np.array(result_method)[:, 1]) + 0.25)
            plt.figure(figsize=(7, 6))
            
            #Vytvoenie mriežky:
            x1 = np.linspace(*x1_range, 100)
            x2 = np.linspace(*x2_range, 100)
            X1, X2 = np.meshgrid(x1, x2)
            
            #Určenie funkčnej hodnoty pre body z mriežky
            X3 = np.zeros_like(X1)
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    xij = np.array([X1[i, j], X2[i, j]])
                    X3[i, j] = f_num(xij)
            
            #Vytvorenie farebnej škály:
            contour = plt.contour(X1, X2, X3, levels=np.linspace(np.min(X3), np.max(X3), 30), cmap="viridis")
            
            #Vizualizácia konvergenčnej cesty na vrstevnicovej mape:
            plt.plot(np.array(result_method)[:, 0], np.array(result_method)[:, 1], marker="o", markeredgewidth=0.5, linestyle="dashed", label=method, color = colors[m])
            
            #Vyznačenie bodu minima:
            plt.scatter(result[0], result[1], s=100, c="red", marker="x", label=r"$x_{\text{opt}}$")
            result_method = np.array(result_method)
            
            #Obmedzenie osí:
            plt.xlim(min(result_method[:, 0]) - 0.25, max(result_method[:, 0]) + 0.25)
            plt.ylim(min(result_method[:, 1]) - 0.25, max(result_method[:, 1]) + 0.25)
            plt.xlabel(r"$x_1$", fontsize=14)
            plt.ylabel(r"$x_2$", fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(f"Konvergenčná cesta k optimánemu bodu pre {title} - {method}")
            plt.legend(fontsize=14, framealpha=1)
            
            plt.tight_layout()
            plt.show()



#Vykreslenie 3D grafu a vrstevnicovej mapy konkrétnej funkcie:
    
#Vstupnými parametrami sú: - numerický tvar účelovej funkcie
#                          - skutočné riešenie (skutočný bod minima)
#                          - polomer (okolia od bodu minima)
#                          - názov účelovej funckie (zadáva sa v tvare: title = "Názov funckie" )
def function_plot(f_num, result, radius, title):
    
    #Funkcia Himmelblau má 4 minimá, preto si vyžaduje unikátny prístup pri voľbe rozmedia.
    if title == "Himmelblau":
        x_range = (-5, 5)
        y_range = (-5, 5)
        
    #Iné funkcie majú iba jedno globálne minimum, teda rozmedzie sa prispôsobuje tomuto bodu.
    else:
        x_range = (result[0] - radius, result[0] + radius)
        y_range = (result[1] - radius, result[1] + radius)
        
    #Vytvorenie mriežky:
    x1 = np.linspace(*x_range, 100)
    x2 = np.linspace(*y_range, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = np.zeros_like(X1)
    
    #Určenie funkčnej hodnoty v každom bode mriežky:
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            xij = np.array([X1[i, j], X2[i, j]])
            X3[i, j] = f_num(xij)

    fig = plt.figure(figsize=(12, 5))
    
    #Vytvorenie 3D grafu:
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X1, X2, X3, cmap="viridis")
    ax1.set_xlabel(r"$x_1$", fontsize=14)
    ax1.set_ylabel(r"$x_2$", fontsize=14)
    ax1.set_title("Funkcia " + title)
    
    #Vytvorenie vrstevnicovej mapy:
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X1, X2, X3, levels=np.linspace(np.min(X3), np.max(X3), 30), cmap="viridis")
    fig.colorbar(contour, ax=ax2)
    
    #Pre funkciu Himmelblau treba vyznačiť aj zvyšné tri minimá.
    if title == "Himmelblau":
        for i in range(result.shape[0]):
            label_1 = r"$x_{\text{opt}}$" if i == 0 else ""
            ax2.scatter(result[i][0], result[i][1], s=100, c="red", marker="x", label=label_1)
    else:
        ax2.scatter(result[0], result[1], s=100, c="red", marker="x", label=r"$x_{\text{opt}}$")
    ax2.set_xlabel(r"$x_1$", fontsize=14)
    ax2.set_ylabel(r"$x_2$", fontsize=14)
    ax2.set_title("Vrstevnicová mapa funkcie " + title)
    ax2.legend(fontsize=14, framealpha=1)



#Vykreslenie konvergenčných máp pre metódy DFP, BFGS, SR1 a Hoshino:
#(Funguje to iba pre dvojrozmerné funkcie.)

#Vstupnými parametrami sú: - numerický tvar účelovej funkcie
#                          - symbolický tvar účelovej funckie
#                          - tolerancia
#                          - metóda (zadáva sa názov v úvodzovkách, napr. "BFGS" )
#                          - rozmedie x1 (zadáva sa tvare: x1_range = (a, b) )
#                          - rozmedie x2 (zadáva sa tvare: x2_range = (c, d) )
#                          - názov účelovej funckie (zadáva sa v tvare: title = "Názov funckie" )
def convergence_map_full(f_num, f_sym, tol, method, x1_range, x2_range, title):
    
    #Spočítanie gradientu účelovej funkcie:
    g_sym = gradient_sym(f_sym, 2)
    g_num = gradient_num(g_sym, 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    #Nastavenie rozlíšenia:
    resolution = 200
    
    #Vytvorenie mriežky:
    x1 = np.linspace(*x1_range, resolution)
    x2 = np.linspace(*x2_range, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = np.full_like(X1, np.nan, dtype=float)
    
    #V tomto cykle sa každému bodu z mriežky priradí počet iterácií, za ktorý metóda skonvergovala, prípadne NAN v prípade zlyhania.
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            xij = np.array([X1[i, j], X2[i, j]])
            x_k = quasiNewton_method(f_num, f_sym, xij, tol, method)
            if np.linalg.norm(np.array(g_num(*x_k[-1]), dtype=float)) <= tol:
                X3[i, j] = len(x_k) - 1
            else:
                X3[i, j] = np.nan
                
    #Nastavenie pozadia na čiernu:
    ax.set_facecolor("black")
    
    #Nastavenie farebnej škály, prislúchajúcej počtu iterácií:
    X3_no_nan = X3[~np.isnan(X3)]
    if X3_no_nan.size > 0:
        levels = np.arange(int(np.min(X3_no_nan)), int(np.max(X3_no_nan)) + 2)
        contour = ax.contourf(X1, X2, X3, levels=levels, cmap="viridis", extend='both')
        fig.colorbar(contour, ax=ax)
    else:
        print("No points converged.")
    
    #Vykreslenie konvergenčnej mapy:
    ax.set_title(f"Konvergenčná mapa pre funkciu {title}, metóda {method}")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    
    ax.legend([Patch(facecolor="black")], ["Zlyhanie"], fontsize=14, framealpha=1)
    plt.tight_layout()
    plt.show()
    
    
   
#convergence_map_full(f1_num, f1_sym, 1e-6, "Hoshino", x1_range = (-3, 7), x2_range = (-6, 4), title=r"$f_1$")
#convergence_map_full(rosenbrock_num, rosenbrock_sym, 1e-6, "SR1", x1_range = (-2, 2), x2_range = (-2, 2), title="Rosenbrock")
convergence_map_full(himmelblau_num, himmelblau_sym, 1e-6, "DFP", x1_range = (-5, 5), x2_range = (-5, 5), title="Himmelblau")    
#convergence_map_full(zakhar_num, zakhar_sym, 1e-6, "Hoshino", x1_range = (-2, 2), x2_range = (-2, 2), title="Zakhar")
    
#quasiNewton_method(f1_num, f1_sym, x0_2, 1e-6, "BFGS")
#visualize(f1_num, f1_sym, x0_1, result_f1, 1e-6, "$f_1$")
#visualize(rosenbrock_num, rosenbrock_sym, x0_1, result_rosenbrock, 1e-6, "Rosenbrock")
#function_plot(rosenbrock_num, result_rosenbrock, 2, "Rosenbrock")
#quasiNewton_method(test_multi_num, test_multi_sym, x0_4, 1e-6, "BFGS")
#visualize(test_multi_num, test_multi_sym, x0_4, result_test_multi, 1e-6, "$f_2$")
#function_plot(himmelblau_num, result_himmelblau, 5, "Himmelblau")
