import pandas as pd  # Datenanalyse und -manipulation
import statsmodels.api as sm  # Statistische Modelle und Regressionen
import matplotlib.pyplot as plt  # Plot-Erstellung
import seaborn as sns  # Erweiterte Visualisierung
import scipy.stats as stats  # Statistische Tests
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Prüft Multikollinearität
from statsmodels.stats.diagnostic import het_breuschpagan  # Test auf Heteroskedastizität

# 1. Tabelle einlesen
dateipfad = "/home/xstarcroftx/Project/Prinzip_Rate.xlsx"  # Absoluter Pfad zur Excel-Datei
tabelle = pd.read_excel(dateipfad, sheet_name="Prinzip_Rate")  # Liest das Excel-Sheet "Tabelle4" ein

# 2. Features definieren (Cialdini-Prinzipien)
prinzipien_namen = tabelle.columns[0:6]  # Erste 6 Spalten als erklärende Variablen

# 3. Zielvariable definieren
kompromittier_rate = tabelle["Kompromittierrate"]  # Zielvariable: Erfolgsrate der Kompromittierung

# 4. Features extrahieren
prinzipien_daten = tabelle[prinzipien_namen]  # Nur die Prinzipien-Spalten

# 5. VIF-Berechnung zur Multikollinearitätsprüfung
prinzipien_daten_mit_cons = sm.add_constant(prinzipien_daten)  # Konstante (Intercept) hinzufügen
vif_tabelle = pd.DataFrame()
vif_tabelle["Merkmal"] = prinzipien_daten_mit_cons.columns
vif_tabelle["VIF"] = [variance_inflation_factor(prinzipien_daten_mit_cons.values, i) for i in range(prinzipien_daten_mit_cons.shape[1])]
print("--- Variance Inflation Factors (VIF) ---")
print(vif_tabelle)  # VIF > 5 = potenzielle Multikollinearität

# 6. OLS-Modell fitten
modell = sm.OLS(kompromittier_rate, prinzipien_daten_mit_cons).fit()
print("\n--- OLS Regressionsanalyse (nicht robust) ---")
print(modell.summary())  # Modellzusammenfassung inkl. Koeffizienten, R², Signifikanz

# 7. Heteroskedastizität: Breusch-Pagan-Test
bp_test = het_breuschpagan(modell.resid, modell.model.exog)
bp_p_wert = bp_test[1]
print("\n--- Breusch-Pagan-Test auf Heteroskedastizität ---")
print(f"P-Wert: {bp_p_wert}")
if bp_p_wert < 0.05:
    print("Hinweis: Es gibt Hinweise auf Heteroskedastizität.")
else:
    print("Hinweis: Keine Hinweise auf Heteroskedastizität.")

# 8. Robustere Standardfehler bei Heteroskedastizität (HC3)
robustes_modell = sm.OLS(kompromittier_rate, prinzipien_daten_mit_cons).fit(cov_type='HC3')
print("\n--- OLS mit robusten Standardfehlern (HC3) ---")
print(robustes_modell.summary())

# 9. Visualisierungen
sns.set(style="whitegrid")

# Residuen vs. Fitted Plot
plt.figure(figsize=(8, 5))
sns.residplot(x=modell.fittedvalues, y=modell.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Vorhergesagte Werte")
plt.ylabel("Residuen")
plt.title("Residuen vs. Vorhergesagte Werte")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig("Residuen vs. Fitted Plot.png")

# Histogramm der Residuen
plt.figure(figsize=(8, 4))
sns.histplot(modell.resid, kde=True, bins=20)
plt.title("Histogramm der Residuen")
plt.xlabel("Residuen")
plt.ylabel("Häufigkeit")
plt.tight_layout()
plt.savefig("Histogramm.png")

# QQ-Plot der Residuen
sm.qqplot(modell.resid, line='45')
plt.title("QQ-Plot der Residuen")
plt.tight_layout()
plt.savefig("QQ-Plot der Residuen.png")

# Scatterplots mit Regressionslinien für jedes Prinzip
feature = prinzipien_daten.iloc[:, 0]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[0])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[0]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Reziprozität.png")

feature = prinzipien_daten.iloc[:, 1]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[1])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[1]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Verpfl. & Konsistenz.png")

feature = prinzipien_daten.iloc[:, 2]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[2])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[2]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Sozl. Bewährtheit.png")

feature = prinzipien_daten.iloc[:, 3]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[3])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[3]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Sympathie.png")

feature = prinzipien_daten.iloc[:, 4]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[4])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[4]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Autorität.png")

feature = prinzipien_daten.iloc[:, 5]
plt.figure(figsize=(8, 5))
sns.regplot(x=feature, y=kompromittier_rate, line_kws={"color": "red"})
plt.xlabel(prinzipien_namen[5])
plt.ylabel("Kompromittierrate")
plt.title(f"Lineare Regression: {prinzipien_namen[5]} vs. Kompromittierrate")
plt.tight_layout()
plt.savefig("Scatterplot Knappheit.png")
