import pandas as pd                                                                                                                     # Importiert pandas, eine Bibliothek zur Datenmanipulation und -analyse
import scipy.stats as stats                                                                                                             # Importiert scipy.stats für statistische Tests und Verteilungen
import itertools                                                                                                                        # Importiert itertools für das Erstellen von Iteratoren und Kombinationen
import numpy as np                                                                                                                      # Importiert numpy für numerische Berechnungen
from statsmodels.stats.multitest import multipletests                                                                                   # Importiert die Funktion multipletests für die Durchführung von Korrekturen nach mehreren Tests
import matplotlib.pyplot as plt
import seaborn as sns


# === Excel-Datei einlesen ===
dateipfad = "/home/xstarcroftx/Project/Prinzip_Rate.xlsx"                                                                               # Definiert den Pfad zur Excel-Datei
tabelle = pd.read_excel(dateipfad, sheet_name="Prinzip_Rate")                                                                               # Liest die Excel-Datei ein, speziell das Arbeitsblatt "Tabelle4" in ein DataFrame

# === Cialdini-Prinzipien befinden sich in den ersten 6 Spalten ===
header_spalten = tabelle.columns[0:6]                                                                                                   # Extrahiert die ersten 6 Spalten der Tabelle (die Cialdini-Prinzipien) und speichert sie in der Variablen 'prinzip_spalten'
print("\nCialdini-Prinzipien (Spaltennamen):")                                                                                          # Gibt eine Überschrift für die Ausgabe der Cialdini-Prinzipien aus
# print("\nheader_spalten:\n", header_spalten)                                                                                                                   # Gibt die Namen der ersten 6 Spalten aus, die die Cialdini-Prinzipien repräsentieren


# === Methode befindet sich in Spalte 8 (Index 7) ===
tabelle['Methode'] = tabelle.iloc[:, 7]                                                                                                 # greift auf alle Zeilen (:) der Spalte H(7) zu. Spaltenzuweisung innerhalb eines DataFrames
# print("\ntabelle['Methode']:\n")

# === Alle einzigartigen Methoden extrahieren ===
methoden = tabelle['Methode'].dropna().unique()                                                                                         # dropna() entfernt Nan Werte, unique notwendig weil Zellen der Methodennamen verbunden wurden
# print("\nGefundene Methoden:")
# print("\nmethoden:\n", methoden)

# === Parameter für Blockverarbeitung ===
blockgröße = 30
startzeile = 0

# === Analyse pro Methode (blockweise) ===
for i, methode in enumerate(methoden):                                                                                                  # iteriert durch die Methoden [Achtung Excel Tabelle startet eig bei 1 (reserviert für Header) Werte befinden sich in Zeile 2 = 0 in Python]
    start_index = startzeile + i * blockgröße                                                                                           # Startzeile für Generic Phishing wäre 0 + 0 * 30 = 0; für Spear Phishing 0 + 1 * 30 = 30 (Zeile 2-31 in Excel) 
    end_index = start_index + blockgröße                                                                                                # Endzeile für Generic Phishing wäre 0 + 30 = 30; für Spear Phishing 30 + 30 = 60 (Zeile 32-61 in Excel)
    block = tabelle.iloc[start_index:end_index]                                                                                         # iloc extrahiert die Zeilen von Start bis Endindex (also Generic Phishing, Spear Phishing etc.)

    print(f"\n--- Block {i+1}: Methode '{methode}' (Zeilen {start_index+2}-{end_index+1}) ---")                                         # Block {i+1} verhindert, dass Block 0 steht, {methode} gibt aktuelle Methode aus; {start_index+2}-{end_index+1} gibt korrekte Excel Zeile aus

    # === Daten für den aktuellen Block anzeigen ===
    daten = block[header_spalten].to_numpy()    
    # print("\ndaten:\n", daten)
                                                                                                                                        # Extrahiert die Werte der angegebenen Prinzipien-Spalten aus dem Block als NumPy-Array                                                                                                                     # Gibt die extrahierten Daten aus dem Block als NumPy-Array aus

    # === Spalten mit konstanter Ausprägung herausfiltern ===
    nicht_konstante_spalten = []                                                                                                        # Liste für die Spalten mit Variation (mehr als einen einzigartigen Wert)
    for spalte_index in range(daten.shape[1]):                                                                                          # Iteriert durch alle Spalten im NumPy-Array (daten) von 0 bis zur letzten Spalte (=.shape[1])
        einzigartige_werte = set(daten[:, spalte_index])                                                                                # Extrahiert alle einzigartigen Werte der aktuellen Spalte; set entfernt doppelte Werte
        # print(f"Spalte {header_spalten[spalte_index]} – Einzigartige Werte: {einzigartige_werte}")                                    # Gibt die einzigartigen Werte der aktuellen Spalte aus
        if len(einzigartige_werte) > 1:                                                                                                 # Wenn es mehr als einen einzigartigen Wert gibt, wird diese Spalte weiter betrachtet
            nicht_konstante_spalten.append(spalte_index)                                                                                # Diese Spalte hat Variation, also wird sie zur Liste 'nicht_konstante_spalten' hinzugefügt
        elif len(einzigartige_werte) == 1:                                                                                              # Wenn die Spalte nur einen einzigartigen Wert hat
            wert = list(einzigartige_werte)[0]                                                                                          # Der einzige Wert der Spalte wird extrahiert
            spaltenname = header_spalten[spalte_index]                                                                                  # Der Name der Spalte wird extrahiert
            if wert == 0:  # Wenn der Wert 0 ist
                print(f"Hinweis: Das Prinzip '{spaltenname}' wurde konstant mit 0 bewertet – möglicherweise nicht relevant.")           # Hinweis, dass das Prinzip konstant mit 0 bewertet wurde
            elif wert == 5:  # Wenn der Wert 5 ist
                print(f"Hinweis: Das Prinzip '{spaltenname}' wurde konstant mit 5 bewertet – möglicherweise besonders relevant.")       # Hinweis, dass das Prinzip konstant mit 5 bewertet wurde


    # DEBUG: Spalten mit Variation anzeigen
    # print("\nSpalten mit mehr als einem einzigartigen Wert:")
    # print("\nnicht_konstante_spalten:\n", nicht_konstante_spalten)  # Gibt die Indizes der Spalten mit Variation aus

    # === Werte für den Friedman-Test extrahieren ===
    werte_friedman = [daten[:, i] for i in nicht_konstante_spalten]                                                                     # Extrahiert die Spalten aus 'daten', die Variation enthalten
    # print("\nwerte_friedman:\n", werte_friedman)

    

    # DEBUG: Werte für den Friedman-Test anzeigen
    # print("\nDaten für den Friedman-Test (ausgewählte Spalten):")
    # print(werte_friedman)  # Gibt die Daten für den Friedman-Test aus

    # === Friedman-Test durchführen ===
    print("\nFühre Friedman-Test durch...")
    statistik, p_wert = stats.friedmanchisquare(*werte_friedman)                                                                        # Führt den Friedman-Test durch und gibt Statistik und p-Wert zurück; stats = Bibliothek scipy.stats, friedmanchisquare = Funktion zur Berechnung des Friedmantests die idR einen Array erwartet, * = * = entpackt die Liste 'werte_friedman' (Spalten aus nicht_konstante_spalten), sodass jede Spalte als eigenes Argument übergeben wird.
    print(f"Friedman-Test Ergebnis: Statistik = {statistik:.4f}, p-Wert = {p_wert:.4f}")                                                # gibt Statistik- und p-Wert gerundet auf 4 Dezimalstellen als Fließkommazahl aus


    # === Post-hoc Wilcoxon-Tests (mit Bonferroni-Korrektur) ===
    p_werte = []                                                                                                                        # Liste für die p-Werte der Wilcoxon-Tests
    paarvergleiche = []                                                                                                                 # Liste für die durchgeführten Paarvergleiche
    teststatistiken = {}                                                                                                                # Dictionary für die Teststatistiken der Paarvergleiche
    mediane = {}                                                                                                                        # Dictionary für die Mediane der Spalten

    # === Median je Prinzip berechnen ===
    for spalte_index in nicht_konstante_spalten:                                                                                        # Iteriert durch alle Spalten mit Variation
        werte = daten[:, spalte_index]                                                                                                  # Holt alle Werte der aktuellen Spalte
        werte_sortiert = np.sort(werte)                                                                                                 # Sortiert die Werte aufsteigend
        # print(f"Werte für {header_spalten[spalte_index]} (sortiert): {werte_sortiert}")                                               # Gibt die sortierten Werte aus

        median_wert = np.median(daten[:, spalte_index])                                                                                 # Berechnet den Median der aktuellen Spalte
        spaltenname = header_spalten[spalte_index]                                                                                      # Holt den Namen der Spalte
        mediane[spaltenname] = median_wert                                                                                              # Speichert den Median für die Spalte im Dictionary 'mediane' (s. Zeile 80); dictionary[key]=value == mediane[Reziprozität]:2,5
        # print(f"Median von {spaltenname}: {median_wert}")                                                                               # Gibt den Median der aktuellen Spalte aus

    # === Alle Paarvergleiche (Wilcoxon) durchführen ===
    for i1, i2 in itertools.combinations(nicht_konstante_spalten, 2):                                                                   # Erzeugt alle möglichen Paarvergleiche von nicht konstanten Spalten
        name1 = header_spalten[i1]                                                                                                      # Name der ersten Spalte im Paarvergleich
        name2 = header_spalten[i2]                                                                                                      # Name der zweiten Spalte im Paarvergleich
        print(f"\nVergleich: {name1} vs {name2}")                                                                                       # Gibt an, welche beiden Spalten verglichen werden
        try:
            # print(f"Wertevergleich {name1} vs {name2}:")
            # print(f"{name1}: {daten[:, i1].tolist()}")
            # print(f"{name2}: {daten[:, i2].tolist()}")
            stat, p = stats.wilcoxon(daten[:, i1], daten[:, i2])                                                                        # Führt den Wilcoxon-Test für die beiden Spalten durch
            print(f"Wilcoxon Ergebnis: Statistik = {stat:.4f}, p-Wert = {p:.4f}")                                                       # Gibt das Ergebnis des Wilcoxon-Tests aus
            paarvergleiche.append((i1, i2))                                                                                             # Speichert das Paar der verglichenen Spalten
            p_werte.append(p)                                                                                                           # Speichert den p-Wert des Tests
            teststatistiken[(i1, i2)] = stat                                                                                            # Speichert die Teststatistik
        except ValueError as e:                                                                                                         # Falls ein Fehler beim Wilcoxon-Test auftritt (z.B. bei konstanten Werten)
            print(f"Fehler bei Wilcoxon-Test: {e}")                                                                                     # Gibt den Fehler aus
            continue                                                                                                                    # Überspringt diesen Vergleich und fährt mit dem nächsten fort

    # === Bonferroni-Korrektur durchführen ===
    print("\nFühre Bonferroni-Korrektur durch...")                                                                                      # Gibt an, dass die Bonferroni-Korrektur durchgeführt wird
    ablehnung, korrigierte_p, _, _ = multipletests(p_werte, method='bonferroni')                                                        # Führt die Bonferroni-Korrektur auf die p-Werte durch
    # print("Korrigierte p-Werte:", korrigierte_p)                                                                                      # Gibt die korrigierten p-Werte aus
    # print("Signifikante Vergleiche:", ablehnung)                                                                                      # Gibt an, welche Vergleiche signifikant sind

    # === Signifikante Ergebnisse ausgeben ===
    if any(ablehnung):                                                                                                                  # Wenn es signifikante Vergleiche gibt
        print("\nSignifikante Unterschiede nach Bonferroni-Korrektur:")                                                                 # Gibt an, dass signifikante Unterschiede gefunden wurden
        for idx, ist_signifikant in enumerate(ablehnung):                                                                               # Iteriert durch alle signifikanten Vergleiche
            if ist_signifikant:                                                                                                         # Wenn der Vergleich signifikant ist
                # Holt die Indizes des signifikanten Paars
                i1, i2 = paarvergleiche[idx]                                                                                            # Hole die Spaltenindizes des betroffenen Prinzipienpaars (z. B. Reziprozität vs. Knappheit)
            
                # Ermittle die Namen der Prinzipien (Spaltennamen) anhand ihrer Indizes
                name1 = header_spalten[i1]
                name2 = header_spalten[i2]
            
                # Abrufen der Teststatistik für das Prinzipienpaar (z. B. Wilcoxon-Wert)
                stat = teststatistiken[(i1, i2)] 
            
                # Abrufen des Bonferroni-korrigierten p-Werts für das Paar
                p_korr = korrigierte_p[idx]  
            
                # Mediane der beiden Prinzipien über alle 30 Fälle hinweg (für Vergleich der Effektstärken)
                median1 = mediane[name1]  
                median2 = mediane[name2]

                # Ausgabe der Zwischenschritte zur Nachverfolgung
                # print(f"\nVergleich zwischen {name1} und {name2}:")
                # print(f"  Indizes: i1 = {i1}, i2 = {i2}")
                # print(f"  Median von {name1}: {median1}")
                # print(f"  Median von {name2}: {median2}")
                # print(f"  Teststatistik: {stat:.4f}")
                # print(f"  Korrigierter p-Wert: {p_korr:.4f}")    

                # Bestimmt, welches Prinzip stärker ist (höheres Median)
                if median1 > median2:                                                                                                       # Vergleicht, ob das erste Prinzip einen höheren Median hat als das zweite
                    staerker, schwaecher = name1, name2                                                                                     # Wenn das erste Prinzip stärker ist, wird es als 'staerker' bezeichnet, das zweite als 'schwaecher'
                    median_staerker, median_schwaecher = median1, median2                                                                   # Die entsprechenden Mediane werden den Variablen zugewiesen
                else:                                                                                                                       # Wenn das zweite Prinzip einen höheren Median hat
                    staerker, schwaecher = name2, name1                                                                                     # Das zweite Prinzip wird als 'staerker' bezeichnet, das erste als 'schwaecher'
                    median_staerker, median_schwaecher = median2, median1                                                                   # Die Mediane werden den Variablen zugewiesen

                # Gibt das signifikante Ergebnis aus
                print(f"'{staerker}' ist signifikant stärker als '{schwaecher}' (p = {p_korr:.4f}), "
                      f"Median {staerker}: {median_staerker:.2f} vs. {schwaecher}: {median_schwaecher:.2f}.")

    else:
        print("\nKeine signifikanten Unterschiede nach Bonferroni-Korrektur.")                                                              # Gibt an, dass es keine signifikanten Unterschiede gab

# Leere DataFrame für die Heatmap-Daten vorbereiten
heatmap_data = pd.DataFrame(index=methoden, columns=header_spalten, dtype=float)

# Setze Mediane je Methode und Prinzip in die Heatmap-Tabelle
for i, methode in enumerate(methoden):
    start_index = startzeile + i * blockgröße
    end_index = start_index + blockgröße
    block = tabelle.iloc[start_index:end_index]
    daten = block[header_spalten].to_numpy()

    for j, spaltenname in enumerate(header_spalten):
        # Immer Median setzen, auch bei konstanten Werten
        heatmap_data.loc[methode, spaltenname] = np.median(daten[:, j])
        
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Median-Bewertung'})
plt.title("Relevanz der Cialdini-Prinzipien je Methode (Median)")
plt.xlabel("Cialdini-Prinzipien")
plt.ylabel("Methoden")
plt.tight_layout()
plt.savefig("heatmap_cialdini_methoden.png", dpi=300, bbox_inches='tight')