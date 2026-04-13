# Goodvibes

Python implementation of the goodvibes model for macromolecular diffuse scattering.

## Contributing

### Conventions

- The project is structured as a pip-installable python package similar to mdx2.

- CLI syntax similar to DIALS and mdx2, like: `gv.version`, or `gv.dev.fancy_algorithm` for features in development.

- Developers should install pytest and ruff, and apply these tools for all code added to `src/`.

### Git workflow

Use this to get started (may be revised later):

1. Create an issue on Github. Usually this will take the form of a bugfix or a feature request.

1. Create a new branch off of `dev`. If the branch addresses a particular issue (preferred), name it according to this template: `<your_initials>/<issue_number>-<short_description>`. For instance, `spm/123-new-feature`. If the branch does not address a particular issue, omit the issue number.

1. Commit changes to the feature branch. Push regularly to github. This is good practice if you want to get feedback or you are working on multiple machines.

1. When you are finished, rebase your changes onto the current `dev` branch, and then submit a pull request. Tag the issue number ("resolves issue \#123"). Commits shall not be squashed. If the changes require code review (optional), flag a team member in github and send them a message on Slack.

1. To bring changes into the `main` branch, first do the PR `feature-branch --> dev` as described above, then do a PR of `dev- -> main`. PRs into the main branch should be code reviewed. The commit history shall be squashed (keeps the history clean for `main`).

1. Hotfixes to `main` are possible, but require some care to maintain a clean history.

### Prototyping

- For early-stage prototyping, create a directory for yourself in `sandbox/`. Files in this folder are not part of the goodvibes package (they cannot be imported when goodvibes in pip-installed), but they are included with the repo. Keeping prototype code confined to your sandbox folder makes it easier to merge feature branches into dev, as conflicts are avoided.

- Please do not add large data files to your folder (use backblaze or other shared storage for that -- same goes for testing fixtures).