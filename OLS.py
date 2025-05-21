import pandas as pd                                                                                                                         # Importiert die pandas-Bibliothek für die Datenmanipulation
import statsmodels.api as sm  
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats                                                                                                                 # Importiert das 'statsmodels'-Modul für statistische Modelle
from statsmodels.stats.outliers_influence import variance_inflation_factor                                                                  # Importiert die Funktion zur Berechnung des VIF (Variance Inflation Factor)
from statsmodels.stats.diagnostic import het_breuschpagan                                                                                   # Importiert den Breusch-Pagan-Test zur Prüfung von Heteroskedastizität


# 1. Tabelle einlesen
dateipfad = "/home/xstarcroftx/Project/Prinzip_Rate.xlsx"                                                                                   # Pfad zur Excel-Datei
tabelle = pd.read_excel(dateipfad, sheet_name="Prinzip_Rate")                                                                               # Liest das Excel-Blatt "Tabelle4" in ein DataFrame ein

# 2. Relevante Features definieren (die 6 Prinzipien)
prinzipien_namen = tabelle.columns[0:6]                                                                                                     # Definiert die Header (Spaltennamen), die im Modell verwendet werden
# print("\nprinzipien_namen:\n", prinzipien_namen)

# 3. Zielvariable definieren (Kompromittiertrate)
kompromittier_rate = tabelle["Kompromittierrate"]                                                                                           # Definiert die Zielvariable 'Kompromittierrate' aus dem DataFrame
# print("\nkompromittier_rate:\n", kompromittier_rate)

# 4. Daten für Features vorbereiten (inklusive Standardisierung)
prinzipien_daten = tabelle[prinzipien_namen]                                                                                                # Extrahiert die Spalten, die die Features (Prinzipien) enthalten, aus dem DataFrame
# print("\nprinzipien_daten:\n", prinzipien_daten)

# 5. Multikollinearität prüfen - VIF (Variance Inflation Factor)
prinzipien_daten_mit_cons = sm.add_constant(prinzipien_daten)                                                                               # Fügt eine Konstante (Intercept) zu den Features hinzu, um die Multikollinearität zu prüfen
# print("\nprinzipien_daten_mit_cons:\n", prinzipien_daten_mit_cons)

# print(prinzipien_daten_mit_cons)
vif_tabelle = pd.DataFrame()                                                                                                                # Erzeugt ein leeres DataFrame für die VIF-Werte
vif_tabelle["Merkmal"] = prinzipien_daten_mit_cons.columns                                                                                  # Fügt die Spaltennamen (Features) zu diesem DataFrame hinzu
vif_tabelle["VIF"] = [variance_inflation_factor(prinzipien_daten_mit_cons.values, i) for i in range(prinzipien_daten_mit_cons.shape[1])]    # Berechnet den VIF für jedes Feature, .values wandelt DataFrame in NumPy-Array um; range(shape[1]) zählt nur Spaltenanzahl =7 (6 Prinzipien + cons)


print("--- Variance Inflation Factors (VIF) ---")                                                                                           # Gibt eine Überschrift aus, um die VIF-Werte anzuzeigen
print(vif_tabelle)                                                                                                                          # VIF-Wert zeigt die Korrelation zwischen den Features: 1 bedeutet Unabhängigkeit, Werte zwischen 1 und 5 sind unbedenklich, Werte über 5 können die Regression verzerren. (const Wert kann ignoriert werden, da konstante 1 selbsterklärend ist)


# 6. OLS-Modell fitten
modell = sm.OLS(kompromittier_rate, prinzipien_daten_mit_cons).fit()                                                                        # Passt ein OLS (Ordinary Least Squares) Modell an die Daten an
# print("\nmodell:\n", modell)

# print("\n--- Regressionskoeffizienten (inkl. Intercept) ---")
# print(modell.params)

# print("\n--- Bestimmtheitsmaß R² ---")
# print(modell.rsquared)

# print("\n--- P-Werte der einzelnen Variablen ---")
# print(modell.pvalues)

# print("\n--- Residuen (Vorhersagefehler) ---")
# print(modell.resid)


# 7. Modellzusammenfassung (Ergebnisse)
print("\n--- OLS Regressionsanalyse (nicht robust) ---")                                                                                    # Gibt eine Überschrift aus, um das Modell mit robusten Standardfehlern zu kennzeichnen
print(modell.summary())                                                                                                                     # Gibt eine vollständige Zusammenfassung der OLS-Modellergebnisse aus

# 8. Heteroskedastizität prüfen - Breusch-Pagan-Test
# Der Breusch-Pagan-Test prüft auf heteroskedastische Fehler (d.h. Fehler haben unterschiedliche Varianzen)
bp_test = het_breuschpagan(modell.resid, modell.model.exog)                                                                                 # Führt den Breusch-Pagan-Test auf die Residuen des Modells aus
# print("\nmodell.resid:\n", modell.resid)
# print("\nmodell.model.exog:\n", modell.model.exog)

bp_p_wert = bp_test[1]                                                                                                                      # Extrahiert den p-Wert des Tests (der relevante Wert für die Heteroskedastizitätsprüfung)
# print("\nbp_test:\n", bp_test)
print("\n--- Breusch-Pagan-Test auf Heteroskedastizität ---")                                                                               # Gibt eine Überschrift für die Ausgabe des Tests aus
print(f"P-Wert: {bp_p_wert}")  # Gibt den p-Wert des Breusch-Pagan-Tests aus

# 9. Wenn der P-Wert des Breusch-Pagan-Tests < 0.05 ist, gibt es Hinweise auf Heteroskedastizität
if bp_p_wert < 0.05:                                                                                                                        # Wenn der p-Wert kleiner als 0.05 ist, deutet dies auf Heteroskedastizität hin
    print("Hinweis: Es gibt Hinweise auf Heteroskedastizität.")                                                                             # Gibt eine Warnung aus, wenn Heteroskedastizität festgestellt wird
else:                                                                                                                                       # Wenn der p-Wert größer oder gleich 0.05 ist, gibt es keine Hinweise auf Heteroskedastizität
    print("Hinweis: Keine Hinweise auf Heteroskedastizität.")                                                                               # Gibt aus, dass keine Heteroskedastizität vorliegt

# 10. Robustere Standardfehler bei Heteroskedastizität (HC3)
# Wenn Heteroskedastizität vorliegt, sind robuste Standardfehler nützlich
robustes_modell = sm.OLS(kompromittier_rate, prinzipien_daten_mit_cons).fit(cov_type='HC3')                                                 # Passt ein OLS-Modell an und berechnet robuste Standardfehler (HC3)
# print("\nrobustes_modell:\n" ,robustes_modell)

print("\n--- OLS mit robusten Standardfehlern (HC3) ---")                                                                                   # Gibt eine Überschrift für die Ausgabe des robusten Modells aus
print(robustes_modell.summary())                                                                                                            # Gibt eine Zusammenfassung der Ergebnisse des robusten OLS-Modells aus

###################################

# Stil für die Diagramme setzen (weißer Hintergrund mit Rasterlinien)
sns.set(style="whitegrid")

# 1. Residuen vs. Fitted Plot
plt.figure(figsize=(8, 5))                                                                                                                  # Neue Abbildung mit Größe 8x5 Zoll erstellen
# Residuen gegen vorhergesagte Werte plotten (Lowess = geglättete Linie)
sns.residplot(x=modell.fittedvalues, y=modell.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Vorhergesagte Werte")                                                                                                           # X-Achsenbeschriftung setzen
plt.ylabel("Residuen")                                                                                                                      # Y-Achsenbeschriftung setzen
plt.title("Residuen vs. Vorhergesagte Werte")                                                                                               # Diagrammtitel setzen
plt.axhline(0, color='gray', linestyle='--')                                                                                                # Horizontale Linie bei y=0 als Referenz
plt.tight_layout()                                                                                                                          # Layout optimieren (verhindert Überlappungen)
plt.savefig("Residuen vs. Fitted Plot.png")                                                                                                 # Plot als PNG speichern

# 2. Histogramm der Residuen
plt.figure(figsize=(8, 4))                                                                                                                  # Neue Abbildung mit Größe 8x4 Zoll
# Histogramm der Residuen mit KDE (Dichtekurve)
sns.histplot(modell.resid, kde=True, bins=20)
plt.title("Histogramm der Residuen")                                                                                                        # Titel setzen
plt.xlabel("Residuen")                                                                                                                      # X-Achsenbeschriftung
plt.ylabel("Häufigkeit")                                                                                                                    # Y-Achsenbeschriftung
plt.tight_layout()                                                                                                                          # Layout anpassen
plt.savefig("Histogramm.png")                                                                                                               # Diagramm speichern

# 3. QQ-Plot der Residuen
sm.qqplot(modell.resid, line='45')                                                                                                          # QQ-Plot (Quantile-Quantile), line='45' = Referenzlinie
plt.title("QQ-Plot der Residuen")                                                                                                           # Titel setzen
plt.tight_layout()                                                                                                                          # Layout anpassen
plt.savefig("QQ-Plot der Residuen.png")                                                                                                     # Plot speichern

# 5. Optional: Scatterplots mit Regressionslinien für einzelne Prinzipien
# >>> Nur aktivieren, wenn du EIN Prinzip gezielt untersuchen willst <<<

# Feature 1: Erstes Prinzip
feature = prinzipien_daten.iloc[:, 0]                                                                                                       # Erste Spalte aus dem Prinzipien-Datensatz auswählen
plt.figure(figsize=(8, 5))                                                                                                                  # Neue Abbildung
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})                                                                     # Scatterplot + Regressionslinie
plt.xlabel(prinzipien_namen[0])                                                                                                             # X-Achse beschriften mit Prinzip-Name
plt.ylabel("Kompromittierrate")                                                                                                             # Y-Achse beschriften
plt.title(f"Lineare Regression: {prinzipien_namen[0]} vs. Kompromittierrate")                                                               # Titel dynamisch setzen
plt.tight_layout()                                                                                                                          # Layout anpassen
plt.savefig("Scatterplot Reziprozität.png")                                                                                                 # Speichern

# Feature 2: Zweites Prinzip
feature = prinzipien_daten.iloc[:, 1]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[1])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[1]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Verpfl. & Konsistenz.png")

# Feature 3: Drittes Prinzip
feature = prinzipien_daten.iloc[:, 2]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[2])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[2]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Sozl. Bewährtheit.png")

# Feature 4: Viertes Prinzip
feature = prinzipien_daten.iloc[:, 3]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[3])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[3]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Sympathie.png")

# Feature 5: Fünftes Prinzip
feature = prinzipien_daten.iloc[:, 4]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[4])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[4]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Autorität.png")

# Feature 6: Sechstes Prinzip
feature = prinzipien_daten.iloc[:, 5]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[5])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[5]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Knappheit.png")