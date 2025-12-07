from app.copy_gen import generate_evo_copy, save_copy_to_file


def main():
    campaign_name = "EVO Soda Summer Launch"
    brief = (
        "Summer 2025 launch for EVO Soda. Healthy, low-sugar soda with bold, fruity flavors. "
        "Target: health-conscious adults 25-40 who still love the fun of soda but want cleaner ingredients. "
        "Tone: upbeat, modern, confident, not preachy."
    )

    print("Generating copy for:", campaign_name)
    copy_data = generate_evo_copy(brief)

    print("\nHeadlines:")
    for h in copy_data["headlines"]:
        print("  -", h)

    print("\nCTAs:")
    for c in copy_data["ctas"]:
        print("  -", c)

    path = save_copy_to_file(campaign_name, copy_data)
    print(f"\nSaved copy to: {path.resolve()}")


if __name__ == "__main__":
    main()
