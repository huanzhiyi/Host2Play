from recaptchaSolver import solver

if __name__ == "__main__":
    print(solver("https://google.com/recaptcha/api2/demo", proxy=None, verbose=True, headless=False))
