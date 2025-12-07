from app.campaign_generator import generate_campaign


def main():
    campaign_name = "EVO Soda Summer Launch"
    brief = (
        "Summer 2025 launch for EVO Soda. Healthy, low-sugar soda with bold, fruity flavors. "
        "Target: health-conscious adults 25-40 who still love the fun of soda but want cleaner ingredients. "
        "Tone: upbeat, modern, confident, fun, not preachy."
    )

    result = generate_campaign(campaign_name, brief)

    print("\nDone. Assets:")
    print("  Campaign dir:", result["campaign_dir"].resolve())
    print("  Copy file:   ", result["copy_path"].resolve())
    print("  Hero image:  ", result["hero_path"].resolve())
    print("  Metadata:    ", result["metadata_path"].resolve())


if __name__ == "__main__":
    main()
