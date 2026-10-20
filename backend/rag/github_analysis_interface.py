class GitHubAnalysisInterface:

    def __init__(
        self,
        analysis_service
    ):
        self.analysis_service = (
            analysis_service
        )

    def show_menu(self):

        print("\n")
        print("=" * 70)
        print("             GITHUB ANALYSIS MENU")
        print("=" * 70)

        print(
            "\n1. Repository Summary"
        )

        print(
            "2. Architecture Explanation"
        )

        print(
            "3. Summary + Architecture"
        )

        print(
            "4. Exit"
        )

    def run(self):

        while True:

            self.show_menu()

            choice = input(
                "\nChoose an option: "
            ).strip()

            # -----------------------------------------
            # Summary
            # -----------------------------------------

            if choice == "1":

                print("\n")
                print(
                    "=" * 70
                )

                print(
                    "             REPOSITORY SUMMARY"
                )

                print(
                    "=" * 70
                )

                try:

                    summary = (
                        self.analysis_service
                        .summarize()
                    )

                    print("\n")
                    print(summary)

                except Exception as error:

                    print(
                        "\n❌ Summary generation failed:"
                    )

                    print(error)

            # -----------------------------------------
            # Architecture
            # -----------------------------------------

            elif choice == "2":

                print("\n")
                print(
                    "=" * 70
                )

                print(
                    "          ARCHITECTURE EXPLANATION"
                )

                print(
                    "=" * 70
                )

                try:

                    architecture = (
                        self.analysis_service
                        .explain_architecture()
                    )

                    print("\n")
                    print(architecture)

                except Exception as error:

                    print(
                        "\n❌ Architecture analysis failed:"
                    )

                    print(error)

            # -----------------------------------------
            # Both
            # -----------------------------------------

            elif choice == "3":

                print("\n")
                print(
                    "=" * 70
                )

                print(
                    "             REPOSITORY SUMMARY"
                )

                print(
                    "=" * 70
                )

                try:

                    summary = (
                        self.analysis_service
                        .summarize()
                    )

                    print("\n")
                    print(summary)

                except Exception as error:

                    print(
                        "\n❌ Summary generation failed:"
                    )

                    print(error)

                print("\n")
                print(
                    "=" * 70
                )

                print(
                    "          ARCHITECTURE EXPLANATION"
                )

                print(
                    "=" * 70
                )

                try:

                    architecture = (
                        self.analysis_service
                        .explain_architecture()
                    )

                    print("\n")
                    print(architecture)

                except Exception as error:

                    print(
                        "\n❌ Architecture analysis failed:"
                    )

                    print(error)

            # -----------------------------------------
            # Exit
            # -----------------------------------------

            elif choice == "4":

                print(
                    "\nExiting GitHub analysis."
                )

                break

            else:

                print(
                    "\n⚠️ Invalid option. "
                    "Choose 1, 2, 3, or 4."
                )