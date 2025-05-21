import pandas as pd  # Bibliothek zur Datenmanipulation und -analyse
import scipy.stats as stats  # Statistikfunktionen wie der Friedman- und Wilcoxon-Test
import itertools  # Werkzeuge zur effizienten Iteration, z. B. Kombinationen
import numpy as np  # Mathematische Funktionen und Datenstrukturen
from statsmodels.stats.multitest import multipletests  # Korrekturverfahren für multiples Testen
import matplotlib.pyplot as plt  # Visualisierung von Daten (Standard-Plotbibliothek)
import seaborn as sns  # Erweiterung von matplotlib für attraktivere statistische Plots

# === Excel-Datei einlesen ===
dateipfad = "/home/xstarcroftx/Project/Prinzip_Rate.xlsx"  # Pfad zur Excel-Datei
tabelle = pd.read_excel(dateipfad, sheet_name="Prinzip_Rate")  # Lese das Arbeitsblatt "Tabelle4" in ein DataFrame

# === Cialdini-Prinzipien befinden sich in den ersten 6 Spalten ===
header_spalten = tabelle.columns[0:6]  # Speichere die Namen der ersten 6 Spalten (Cialdini-Prinzipien)
# print("\nCialdini-Prinzipien (Spaltennamen):")

# === Methode befindet sich in Spalte 8 (Index 7) ===
tabelle['Methode'] = tabelle.iloc[:, 7]  # Extrahiere die Methodenbezeichnung aus Spalte 8 (Index 7)

# === Alle einzigartigen Methoden extrahieren ===
methoden = tabelle['Methode'].dropna().unique()  # Entferne fehlende Werte und bestimme eindeutige Methodenbezeichnungen

# === Parameter für Blockverarbeitung ===
blockgröße = 30  # Jede Methode umfasst 30 Zeilen
startzeile = 0  # Startindex der ersten Methode

# === Analyse pro Methode (blockweise) ===
for i, methode in enumerate(methoden):
    start_index = startzeile + i * blockgröße  # Berechne Startzeile für die Methode
    end_index = start_index + blockgröße  # Berechne Endzeile für die Methode
    block = tabelle.iloc[start_index:end_index]  # Extrahiere den Datenblock für die aktuelle Methode

    print(f"\n--- Block {i+1}: Methode '{methode}' (Zeilen {start_index+2}-{end_index+1}) ---")

    daten = block[header_spalten].to_numpy()  # Extrahiere die Werte der Prinzipien als NumPy-Array

    # === Spalten mit konstanter Ausprägung herausfiltern ===
    nicht_konstante_spalten = []  # Liste für Spalten mit mehr als einem einzigartigen Wert
    for spalte_index in range(daten.shape[1]):
        einzigartige_werte = set(daten[:, spalte_index])  # Alle unterschiedlichen Werte in der Spalte
        if len(einzigartige_werte) > 1:
            nicht_konstante_spalten.append(spalte_index)
        elif len(einzigartige_werte) == 1:
            wert = list(einzigartige_werte)[0]
            spaltenname = header_spalten[spalte_index]
            if wert == 0:
                print(f"Hinweis: Das Prinzip '{spaltenname}' wurde konstant mit 0 bewertet – möglicherweise nicht relevant.")
            elif wert == 5:
                print(f"Hinweis: Das Prinzip '{spaltenname}' wurde konstant mit 5 bewertet – möglicherweise besonders relevant.")

    # === Werte für den Friedman-Test extrahieren ===
    werte_friedman = [daten[:, i] for i in nicht_konstante_spalten]  # Nur variierende Spalten für den Test verwenden

    # === Friedman-Test durchführen ===
    print("\nFühre Friedman-Test durch...")
    statistik, p_wert = stats.friedmanchisquare(*werte_friedman)  # Test auf Unterschiede zwischen verbundenen Gruppen
    print(f"Friedman-Test Ergebnis: Statistik = {statistik:.4f}, p-Wert = {p_wert:.4f}")

    # === Post-hoc Wilcoxon-Tests (mit Bonferroni-Korrektur) ===
    p_werte = []  # Liste der p-Werte für Paarvergleiche
    paarvergleiche = []  # Indizes der verglichenen Spaltenpaare
    teststatistiken = {}  # Speichere Teststatistiken je Paar
    mediane = {}  # Speichere Mediane je Prinzip

    # === Mediane berechnen ===
    for spalte_index in nicht_konstante_spalten:
        werte = daten[:, spalte_index]
        median_wert = np.median(werte)
        spaltenname = header_spalten[spalte_index]
        mediane[spaltenname] = median_wert

    # === Wilcoxon-Tests durchführen ===
    for i1, i2 in itertools.combinations(nicht_konstante_spalten, 2):
        name1 = header_spalten[i1]
        name2 = header_spalten[i2]
        print(f"\nVergleich: {name1} vs {name2}")
        try:
            stat, p = stats.wilcoxon(daten[:, i1], daten[:, i2])
            print(f"Wilcoxon Ergebnis: Statistik = {stat:.4f}, p-Wert = {p:.4f}")
            paarvergleiche.append((i1, i2))
            p_werte.append(p)
            teststatistiken[(i1, i2)] = stat
        except ValueError as e:
            print(f"Fehler bei Wilcoxon-Test: {e}")
            continue

    # === Bonferroni-Korrektur ===
    print("\nFühre Bonferroni-Korrektur durch...")
    ablehnung, korrigierte_p, _, _ = multipletests(p_werte, method='bonferroni')

    # === Signifikante Ergebnisse ausgeben ===
    if any(ablehnung):
        print("\nSignifikante Unterschiede nach Bonferroni-Korrektur:")
        for idx, ist_signifikant in enumerate(ablehnung):
            if ist_signifikant:
                i1, i2 = paarvergleiche[idx]
                name1 = header_spalten[i1]
                name2 = header_spalten[i2]
                stat = teststatistiken[(i1, i2)]
                p_korr = korrigierte_p[idx]
                median1 = mediane[name1]
                median2 = mediane[name2]

                if median1 > median2:
                    staerker, schwaecher = name1, name2
                    median_staerker, median_schwaecher = median1, median2
                else:
                    staerker, schwaecher = name2, name1
                    median_staerker, median_schwaecher = median2, median1

                print(f"'{staerker}' ist signifikant stärker als '{schwaecher}' (p = {p_korr:.4f}), "
                      f"Median {staerker}: {median_staerker:.2f} vs. {schwaecher}: {median_schwaecher:.2f}.")
    else:
        print("\nKeine signifikanten Unterschiede nach Bonferroni-Korrektur.")

# === Heatmap vorbereiten ===
heatmap_data = pd.DataFrame(index=methoden, columns=header_spalten, dtype=float)  # Leeres DataFrame

# === Mediane je Methode und Prinzip berechnen ===
for i, methode in enumerate(methoden):
    start_index = startzeile + i * blockgröße
    end_index = start_index + blockgröße
    block = tabelle.iloc[start_index:end_index]
    daten = block[header_spalten].to_numpy()

    for j, spaltenname in enumerate(header_spalten):
        heatmap_data.loc[methode, spaltenname] = np.median(daten[:, j])  # Median auch bei konstanten Werten setzen

# === Heatmap erstellen ===
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Median-Bewertung'})
plt.title("Relevanz der Cialdini-Prinzipien je Methode (Median)")
plt.xlabel("Cialdini-Prinzipien")
plt.ylabel("Methoden")
plt.tight_layout()
plt.savefig("heatmap_cialdini_methoden.png", dpi=300, bbox_inches='tight')
