# Gaia Quick Start 🤖

This guide gets Gaia running, step by step. No experience needed — if you can
follow a recipe, you can do this. Read it top to bottom and don't skip steps.

> **What you are building:** a little robot brain that lives in the cloud
> (on Microsoft Azure) and remembers things, searches the web, and chats back.

---

## What you need first

1. **A Microsoft Azure account.** This is where Gaia's brain will live. You can
   [make a free one here](https://azure.microsoft.com/free/). A grown-up's help
   and a credit card may be needed to sign up (the free pieces stay free).
2. **A computer** running Windows (these steps use Windows commands).
3. About **20 minutes**.

That's it. Let's go. 🚀

---

## Step 1 — Put Gaia's brain in the cloud

Click this green button. It opens Azure and sets up everything Gaia needs.

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fthreadkeeper%2Fgaia-robot%2Fmain%2Finfra%2Fazuredeploy.free.json)

This is the **Free / Lite** button — it costs **nothing while you're not using
it**, so it's the safe one to start with.

---

## Step 2 — Fill in the short form

Azure shows a form. You only need to pick three things:

| Box | What to do |
|-----|------------|
| **Subscription** | Pick the one that's already there. |
| **Resource group** | Click *Create new* and type a name, like `gaia`. (Think of this as a labelled box that holds all of Gaia's parts.) |
| **Region** | Pick one close to you, like *East US*. |

Then click **Review + create**, wait for the green check, and click **Create**.

---

## Step 3 — Wait for it to finish

Azure now builds Gaia's brain. This takes a few minutes. ☕

When you see **"Your deployment is complete"**, you're ready for the next step.

---

## Step 4 — Copy Gaia's address

1. On the "deployment complete" page, click **Outputs** on the left.
2. Find the line called **`cosmosEndpoint`**.
3. Click the little copy icon next to it. (This is the web address of Gaia's
   memory. We'll paste it in a moment.)

Keep this somewhere safe for the next step.

---

## Step 5 — Get the code on your computer

Open **PowerShell** (search for it in the Start menu) and type these lines, one
at a time, pressing **Enter** after each:

```powershell
git clone https://github.com/threadkeeper/gaia-robot.git
cd gaia-robot
```

> Don't have `git`? [Install it here](https://git-scm.com/download/win) first,
> then try again.

---

## Step 6 — Tell Gaia its address

Gaia keeps its settings in a file called `.env`. Let's create it:

```powershell
cd infra
Copy-Item .env.sample .env
notepad .env
```

Notepad opens. Find the line that starts with `COSMOS_ENDPOINT=` and paste the
address you copied in **Step 4** right after the `=` sign. Save the file
(**Ctrl + S**) and close Notepad.

---

## Step 7 — Build Gaia's memory boxes

Gaia needs a few "memory boxes" inside the cloud. One small script makes them
all. Still in PowerShell, type:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python cosmos_create.py
```

> Don't have `python`? [Install it here](https://www.python.org/downloads/)
> (tick *"Add Python to PATH"* during install), then try again.

When the script finishes, Gaia's memory is ready. 🎉

---

## Step 8 — Say hello to Gaia

Now start Gaia on your own computer:

```powershell
cd ..
./infra/run-local.ps1
```

This opens Gaia in your web browser. Type a message and watch it think and
reply!

---

## Something went wrong?

- **A red error about "az login" or tokens?** Run `az login` and sign in, then
  try again. ([Install the Azure CLI here](https://aka.ms/installazurecli) if
  the `az` command is missing.)
- **`python` or `git` "not recognized"?** It isn't installed yet — see the
  install links above, then reopen PowerShell.
- **Want the deeper technical details?** See [infra/README.md](infra/README.md).

---

## You did it! 🌟

Gaia is alive. To learn how its brain actually works, head back to the
[main README](README.md#how-gaia-thinks).
