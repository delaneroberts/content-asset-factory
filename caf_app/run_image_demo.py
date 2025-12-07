from app.image_gen import generate_evo_hero_image


def main():
    prompt = (
        "A vibrant, modern hero image for a healthy soda brand called EVO Soda. "
        "Aluminum can with minimalist design, bold 'EVO' lettering, teal and citrus colors, "
        "sparkling bubbles, and a fresh, upbeat summer vibe. 2D digital illustration."
    )

    print("Generating EVO Soda hero image…")
    out_path = generate_evo_hero_image(prompt, filename="evo_soda_hero.png")
    print(f"Saved hero image to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
