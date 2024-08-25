import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

class CorrelationRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Correlation and Regression Calculator of Valeri@ Arriaga Vilchez")

        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 12))
        style.configure('TButton', background='#f0f0f0', font=('Arial', 12))
        style.configure('TCombobox', font=('Arial', 12))

        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.var_count_label = ttk.Label(frame, text="Numero de Variables:")
        self.var_count_label.grid(row=0, column=0, padx=5, pady=5)
        self.var_count_entry = ttk.Entry(frame)
        self.var_count_entry.grid(row=0, column=1, padx=5, pady=5)

        self.submit_button = ttk.Button(frame, text="Submit", command=self.create_entries)
        self.submit_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.regression_type_label = ttk.Label(frame, text="Regression Type:")
        self.regression_type_label.grid(row=2, column=0, padx=5, pady=5)
        self.regression_type = ttk.Combobox(frame, values=["Linear", "Exponential", "Quadratic", "Cubic", "Logarithmic"])
        self.regression_type.grid(row=2, column=1, padx=5, pady=5)
        self.regression_type.current(0)  # set default value

        self.clear_button = ttk.Button(frame, text="Clear All", command=self.clear_all)
        self.clear_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.entries_frame = ttk.Frame(frame, padding="10")
        self.entries_frame.grid(row=4, column=0, columnspan=2)

        self.save_button = ttk.Button(frame, text="Save as SVG", command=self.save_as_svg)
        self.save_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        self.show_correlation_button = ttk.Button(frame, text="Show Correlation Matrix", command=self.show_correlation_matrix)
        self.show_correlation_button.grid(row=6, column=0, padx=5, pady=5)

        self.show_regression_button = ttk.Button(frame, text="Show Regression Matrix", command=self.show_regression_matrix)
        self.show_regression_button.grid(row=6, column=1, padx=5, pady=5)

    def create_entries(self):
        for widget in self.entries_frame.winfo_children():
            widget.destroy()

        self.var_count = int(self.var_count_entry.get())
        self.var_names = []
        self.var_data = []

        for i in range(self.var_count):
            label = ttk.Label(self.entries_frame, text=f"Variable {i+1} Name:")
            label.grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(self.entries_frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.var_names.append(entry)

            data_label = ttk.Label(self.entries_frame, text=f"Variable {i+1} Data (comma separated):")
            data_label.grid(row=i, column=2, padx=5, pady=5)
            data_entry = ttk.Entry(self.entries_frame)
            data_entry.grid(row=i, column=3, padx=5, pady=5)
            self.var_data.append(data_entry)

        self.generate_button = ttk.Button(self.entries_frame, text="Generate Data", command=self.generate_data)
        self.generate_button.grid(row=self.var_count+1, column=0, columnspan=4, padx=5, pady=10)

    def clear_all(self):
        self.var_count_entry.delete(0, tk.END)
        for widget in self.entries_frame.winfo_children():
            widget.destroy()
        self.regression_type.current(0)

    def generate_data(self):
        var_names = [entry.get() for entry in self.var_names]
        data = []
        for entry in self.var_data:
            try:
                data.append(list(map(float, entry.get().split(','))))
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numbers separated by commas.")
                return

        if not all(len(d) == len(data[0]) for d in data):
            messagebox.showerror("Input Error", "All variables must have the same number of data points.")
            return

        self.df = pd.DataFrame(data).T
        self.df.columns = var_names

    def show_correlation_matrix(self):
        if not hasattr(self, 'df'):
            messagebox.showerror("Error", "No data to show.")
            return

        var_names = self.df.columns
        pearson_corr = self.df.corr(method='pearson')
        kendall_corr = self.df.corr(method='kendall')
        spearman_corr = self.df.corr(method='spearman')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Correlation Matrices')

        sns.heatmap(pearson_corr, ax=axes[0], annot=True, cmap='viridis')
        axes[0].set_title('Pearson Correlation')

        sns.heatmap(kendall_corr, ax=axes[1], annot=True, cmap='viridis')
        axes[1].set_title('Kendall Correlation')

        sns.heatmap(spearman_corr, ax=axes[2], annot=True, cmap='viridis')
        axes[2].set_title('Spearman Correlation')

        plt.show()

    def show_regression_matrix(self):
        if not hasattr(self, 'df'):
            messagebox.showerror("Error", "No data to show.")
            return

        regression_type = self.regression_type.get()
        g = sns.PairGrid(self.df, palette="husl")
        
        def linear_regression(x, y, **kwargs):
            x = x.values.reshape(-1, 1)
            reg = LinearRegression().fit(x, y)
            y_pred = reg.predict(x)
            plt.plot(x, y_pred, color='purple')
        
        def exponential_regression(x, y):
            x = x.values
            def exponential_func(x, a, b):
                return a * np.exp(b * x)
            popt, _ = curve_fit(exponential_func, x, y)
            y_pred = exponential_func(x, *popt)
            plt.plot(x, y_pred, color='purple')
        
        def quadratic_regression(x, y):
            x = x.values.reshape(-1, 1)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(x)
            reg = LinearRegression().fit(X_poly, y)
            y_pred = reg.predict(X_poly)
            plt.plot(x, y_pred, color='purple')
        
        def cubic_regression(x, y):
            x = x.values.reshape(-1, 1)
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(x)
            reg = LinearRegression().fit(X_poly, y)
            y_pred = reg.predict(X_poly)
            plt.plot(x, y_pred, color='purple')
        
        def logarithmic_regression(x, y):
            x = x.values
            def logarithmic_func(x, a, b):
                return a * np.log(x) + b
            popt, _ = curve_fit(logarithmic_func, x, y)
            y_pred = logarithmic_func(x, *popt)
            plt.plot(x, y_pred, color='purple')

        def plot_regression(x, y, **kwargs):
            ax = plt.gca()
            ax.scatter(x, y, **kwargs)
            if regression_type == "Linear":
                linear_regression(x, y)
            elif regression_type == "Exponential":
                exponential_regression(x, y)
            elif regression_type == "Quadratic":
                quadratic_regression(x, y)
            elif regression_type == "Cubic":
                cubic_regression(x, y)
            elif regression_type == "Logarithmic":
                logarithmic_regression(x, y)
        
        def plot_distribution(x, **kwargs):
            sns.histplot(x, kde=True, **kwargs)
        
        def plot_equation(x, y, **kwargs):
            ax = plt.gca()
            if regression_type == "Linear":
                x = x.values.reshape(-1, 1)
                reg = LinearRegression().fit(x, y)
                y_pred = reg.predict(x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {reg.intercept_:.2f} + {reg.coef_[0]:.2f}x\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            elif regression_type == "Exponential":
                x = x.values
                popt, _ = curve_fit(lambda t,a,b: a*np.exp(b*t), x.ravel(), y)
                y_pred = popt[0]*np.exp(popt[1]*x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {popt[0]:.2f}e^({popt[1]:.2f}x)\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            elif regression_type == "Quadratic":
                x = x.values.reshape(-1, 1)
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(x)
                reg = LinearRegression().fit(X_poly, y)
                y_pred = reg.predict(X_poly)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {reg.intercept_:.2f} + {reg.coef_[1]:.2f}x + {reg.coef_[2]:.2f}x^2\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            elif regression_type == "Cubic":
                x = x.values.reshape(-1, 1)
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(x)
                reg = LinearRegression().fit(X_poly, y)
                y_pred = reg.predict(X_poly)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {reg.intercept_:.2f} + {reg.coef_[1]:.2f}x + {reg.coef_[2]:.2f}x^2 + {reg.coef_[3]:.2f}x^3\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            elif regression_type == "Logarithmic":
                x = x.values
                popt, _ = curve_fit(lambda t,a,b: a*np.log(t)+b, x.ravel(), y)
                y_pred = popt[0]*np.log(x)+popt[1]
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {popt[0]:.2f}ln(x) + {popt[1]:.2f}\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10, verticalalignment='top')

        g.map_lower(plot_regression)
        g.map_diag(plot_distribution)
        g.map_upper(plot_equation)
        g.fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.show()

    def save_as_svg(self):
        if not hasattr(self, 'df'):
            messagebox.showerror("Error", "No data to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files", "*.svg")])
        if not file_path:
            return

        var_names = self.df.columns
        pearson_corr = self.df.corr(method='pearson')
        kendall_corr = self.df.corr(method='kendall')
        spearman_corr = self.df.corr(method='spearman')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Correlation Matrices')

        sns.heatmap(pearson_corr, ax=axes[0], annot=True, cmap='ocean')
        axes[0].set_title('Pearson Correlation')

        sns.heatmap(kendall_corr, ax=axes[1], annot=True, cmap='ocean')
        axes[1].set_title('Kendall Correlation')

        sns.heatmap(spearman_corr, ax=axes[2], annot=True, cmap='ocean')
        axes[2].set_title('Spearman Correlation')

        plt.savefig(file_path, format='svg')
        plt.close(fig)

        regression_type = self.regression_type.get()
        g = sns.PairGrid(self.df, palette="Paired")
        
        def linear_regression(x, y, **kwargs):
            x = x.values.reshape(-1, 1)
            reg = LinearRegression().fit(x, y)
            y_pred = reg.predict(x)
            plt.plot(x, y_pred, color='purple')
        
        def exponential_regression(x, y):
            x = x.values
            def exponential_func(x, a, b):
                return a * np.exp(b * x)
            popt, _ = curve_fit(exponential_func, x, y)
            y_pred = exponential_func(x, *popt)
            plt.plot(x, y_pred, color='purple')
        
        def quadratic_regression(x, y):
            x = x.values.reshape(-1, 1)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(x)
            reg = LinearRegression().fit(X_poly, y)
            y_pred = reg.predict(X_poly)
            plt.plot(x, y_pred, color='purple')
        
        def cubic_regression(x, y):
            x = x.values.reshape(-1, 1)
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(x)
            reg = LinearRegression().fit(X_poly, y)
            y_pred = reg.predict(X_poly)
            plt.plot(x, y_pred, color='purple')
        
        def logarithmic_regression(x, y):
            x = x.values
            def logarithmic_func(x, a, b):
                return a * np.log(x) + b
            popt, _ = curve_fit(logarithmic_func, x, y)
            y_pred = logarithmic_func(x, *popt)
            plt.plot(x, y_pred, color='purple')

        def plot_regression(x, y, **kwargs):
            ax = plt.gca()
            ax.scatter(x, y, **kwargs)
            if regression_type == "Linear":
                linear_regression(x, y)
            elif regression_type == "Exponential":
                exponential_regression(x, y)
            elif regression_type == "Quadratic":
                quadratic_regression(x, y)
            elif regression_type == "Cubic":
                cubic_regression(x, y)
            elif regression_type == "Logarithmic":
                logarithmic_regression(x, y)
        
        def plot_distribution(x, **kwargs):
            sns.histplot(x, kde=True, **kwargs)
        
        def plot_equation(x, y, **kwargs):
            ax = plt.gca()
            if regression_type == "Linear":
                x = x.values.reshape(-1, 1)
                reg = LinearRegression().fit(x, y)
                y_pred = reg.predict(x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {reg.intercept_:.2f} + {reg.coef_[0]:.2f}x\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            elif regression_type == "Exponential":
                x = x.values
                popt, _ = curve_fit(lambda t,a,b: a*np.exp(b*t), x.ravel(), y)
                y_pred = popt[0]*np.exp(popt[1]*x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {popt[0]:.2f}e^({popt[1]:.2f}x)\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            elif regression_type == "Quadratic":
                x = x.values.reshape(-1, 1)
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(x)
                reg = LinearRegression().fit(X_poly, y)
                y_pred = reg.predict(X_poly)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {reg.intercept_:.2f} + {reg.coef_[1]:.2f}x + {reg.coef_[2]:.2f}x^2\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            elif regression_type == "Cubic":
                x = x.values.reshape(-1, 1)
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(x)
                reg = LinearRegression().fit(X_poly, y)
                y_pred = reg.predict(X_poly)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {reg.intercept_:.2f} + {reg.coef_[1]:.2f}x + {reg.coef_[2]:.2f}x^2 + {reg.coef_[3]:.2f}x^3\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            elif regression_type == "Logarithmic":
                x = x.values
                popt, _ = curve_fit(lambda t,a,b: a*np.log(t)+b, x.ravel(), y)
                y_pred = popt[0]*np.log(x)+popt[1]
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                equation = f'y = {popt[0]:.2f}ln(x) + {popt[1]:.2f}\nMSE: {mse:.2f}\nR^2: {r2:.2f}'
            ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10, verticalalignment='top')

        g.map_lower(plot_regression)
        g.map_diag(plot_distribution)
        g.map_upper(plot_equation)
        g.fig.subplots_adjust(hspace=0.3, wspace=0.3)

        g.savefig(file_path.replace(".svg", "_regression.svg"), format='svg')
        plt.close(g.fig)

if __name__ == "__main__":
    root = tk.Tk()
    app = CorrelationRegressionApp(root)
    root.mainloop()
    
