import pandas as pd
import matplotlib.pyplot as plt

iso = pd.read_csv("work_isothermal_data.csv")
adi = pd.read_csv("work_adiabatic_data.csv")

plt.figure(figsize=(6, 4))
plt.plot(iso["Vf (m^3)"], iso["Work (J)"], label="Isothermal", lw=2)
plt.plot(adi["Vf (m^3)"], adi["Work (J)"], label="Adiabatic", lw=2)

plt.xlabel("Volume (m$_3$)")
plt.ylabel("Work Done(J)")
plt.title("Comparison of Work vs.  Volume")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("homework-4-1/work_comparison.png", dpi=300)
