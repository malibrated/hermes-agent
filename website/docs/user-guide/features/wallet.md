---
sidebar_position: 18
---

# Crypto Wallet

Give your agent its own crypto wallet. Hermes can hold funds, check balances, and send native tokens on Solana and EVM chains — with encrypted key storage and policy-controlled spending limits.

The agent **never has access to private keys**. Keys are encrypted at rest in a local keystore, and every transaction goes through a policy engine that enforces spending limits, rate limits, and owner approval thresholds.

## Installation

The wallet is an optional extra — install what you need:

```bash
# EVM chains (Ethereum, Base, Polygon, Arbitrum, Optimism)
pip install 'hermes-agent[wallet]'

# + Solana support
pip install 'hermes-agent[wallet-solana]'
```

## Setup

### 1. Initialize the encrypted keystore

The keystore holds all your secrets (API keys, wallet private keys) encrypted with a master passphrase.

```bash
hermes keystore init
```

You'll be prompted to create a passphrase. This is needed each time Hermes starts. To avoid typing it every time:

```bash
# Save to your OS credential store (macOS Keychain, Windows Credential Locker,
# GNOME/KDE Secret Service, or Linux keyctl when available)
hermes keystore remember
```

For **headless/Docker/systemd** deployments where no credential store or TTY is available,
you can set `HERMES_KEYSTORE_PASSPHRASE` as an environment variable. This is a conscious
security tradeoff — the passphrase is visible in the process environment — and should only
be used for unattended deployments:

```bash
# Headless/Docker/systemd only — not recommended for interactive use
export HERMES_KEYSTORE_PASSPHRASE="your-passphrase"
```

Hermes intentionally does **not** fall back to a machine-derived encrypted file
for remembered passphrases. In the current same-user execution model, that would
be derivable by the local agent process and would weaken the keystore boundary.
```

### 2. Create a wallet

```bash
# Create a user wallet (all sends require your approval)
hermes wallet create --chain solana

# Or create an agent wallet (auto-approves within policy limits)
hermes wallet create-agent --chain solana --label "Trading Bot"
```

:::tip Fresh wallets recommended
We recommend creating fresh wallets for your agent rather than importing personal wallets. Send the agent some tokens to get started — keep your personal funds separate.
:::

### 3. Fund the wallet

```bash
hermes wallet fund
```

This displays the deposit address. Send tokens to it from your personal wallet or an exchange.

### 4. Enable the wallet toolset

Add `wallet` to your toolsets in `~/.hermes/config.yaml`:

```yaml
toolsets:
- hermes-cli
- wallet
```

Or pass it at runtime:

```bash
hermes chat -t hermes-cli,wallet
```

## Agent Tools

Once the wallet toolset is enabled, the agent gets these tools:

| Tool | Description |
|------|-------------|
| `wallet_list` | List all wallets with addresses and balances |
| `wallet_balance` | Check balance of a specific wallet |
| `wallet_address` | Get a wallet's deposit address (for sharing / receiving) |
| `wallet_send` | Send native tokens — goes through the policy engine |
| `wallet_estimate_gas` | Estimate transaction fees |
| `wallet_history` | View recent transaction history |
| `wallet_networks` | List supported blockchain networks |

The agent can check its own balances, share its address to receive funds, estimate fees, and initiate transfers. It **cannot** read private keys, bypass spending policies, or disable the kill switch.

## CLI Commands

```
hermes wallet create         Create a new wallet (fresh keypair)
hermes wallet create-agent   Create an agent wallet (auto-approve within limits)
hermes wallet import         Import wallet from exported private key (migration)
hermes wallet export         Export private key for migration to another machine
hermes wallet list           List all wallets with balances
hermes wallet balance        Check a wallet's balance
hermes wallet send <to> <amount>   Send tokens (interactive confirmation)
hermes wallet fund           Show deposit address for receiving tokens
hermes wallet history        View transaction history
hermes wallet freeze         Kill switch — block ALL transactions
hermes wallet unfreeze       Resume after freeze
hermes wallet status         Overview of wallet state
```

## Keystore Commands

The keystore manages all encrypted secrets (API keys, wallet keys):

```
hermes keystore init              Create a new encrypted keystore
hermes keystore list              List stored secrets (names only, no values)
hermes keystore set <name>        Add or update a secret
hermes keystore show <name>       Decrypt and display a secret
hermes keystore delete <name>     Remove a secret
hermes keystore set-category      Change a secret's access category
hermes keystore migrate           Import secrets from .env file
hermes keystore remember          Cache passphrase in OS credential store
hermes keystore forget            Remove cached passphrase
hermes keystore change-passphrase Re-encrypt with a new passphrase
hermes keystore audit             Show access log
hermes keystore status            Show keystore status
```

## Security Model

### Encryption

- Master key derived from your passphrase via **Argon2id** (memory-hard KDF, 64MB)
- Each secret encrypted with **XSalsa20-Poly1305** via libsodium SecretBox (AEAD, random nonce per write)
- Master key held in memory only — never written to disk
- Keystore DB file permissions: `0600`, directory: `0700`

### Secret Categories

| Category | Who can access | Examples |
|----------|---------------|----------|
| `injectable` | Agent (via `os.environ`) | API keys — `OPENROUTER_API_KEY` |
| `gated` | Agent on request (logged) | `GITHUB_TOKEN` |
| `sealed` | **Never** the agent | Wallet private keys |
| `user_only` | CLI only | `SUDO_PASSWORD` |

### Policy Engine

Every transaction is evaluated against a configurable set of policies:

| Policy | Description | Default (agent wallet) |
|--------|-------------|----------------------|
| `spending_limit` | Max per transaction | 1.0 native token |
| `daily_limit` | Aggregate daily cap | 5.0 native token |
| `rate_limit` | Max transactions per window | 5 per hour |
| `cooldown` | Minimum time between txns | 30 seconds |
| `require_approval` | Owner approval above threshold | 0.5 native token |
| `allowed_recipients` | Address whitelist | — |
| `blocked_recipients` | Address blacklist | — |

**User wallets** require owner approval for all transactions by default.

**Agent wallets** auto-approve within limits — transactions above the threshold trigger an owner approval prompt.

### Kill Switch

```bash
hermes wallet freeze
```

Instantly blocks all transactions across all wallets. No policy exceptions. Resume with `hermes wallet unfreeze`.

## Transaction Approval

When a transaction requires approval:

**CLI mode:** An interactive prompt appears with the transaction details and approve/deny choices — identical to the dangerous command approval prompt.

**Gateway mode (Telegram/Discord/etc.):** The transaction summary is shown with instructions to reply `/approve` or `/deny`.

## Supported Networks

### Mainnets
- Ethereum (ETH)
- Base (ETH)
- Polygon (POL)
- Arbitrum One (ETH)
- Optimism (ETH)
- Solana (SOL)

### Testnets
- Ethereum Sepolia
- Base Sepolia
- Solana Devnet

### Custom RPC Endpoints

Override default RPC endpoints in `config.yaml`:

```yaml
wallet:
  rpc_endpoints:
    solana: "https://my-custom-rpc.example.com"
    ethereum: "https://eth-mainnet.alchemyapi.io/v2/YOUR_KEY"
```

## Migration Between Machines

To move a wallet to a new machine:

**On the source machine:**
```bash
hermes wallet export --chain solana
# Re-enter your keystore passphrase
# Private key is displayed — copy it securely
```

**On the destination machine:**
```bash
hermes keystore init                     # Set up keystore (if not already done)
hermes wallet import --chain solana --type agent
# Paste the private key when prompted
```

:::warning
The exported private key gives full control of the wallet. Never share it, transmit it over unencrypted channels, or store it in plaintext.
:::

## Configuration

Full wallet configuration in `~/.hermes/config.yaml`:

```yaml
wallet:
  enabled: true
  default_chain: solana

  # Override default RPC endpoints
  rpc_endpoints:
    solana: "https://api.mainnet-beta.solana.com"
    ethereum: "https://eth.llamarpc.com"

  # Minimal policy overrides currently supported at runtime
  # (global/shared state, not per-wallet yet)
  agent_wallet:
    enabled: true
    auto_approve_below_native: "0.5"   # maps to require_approval.above_native
    daily_limit_native: "5.0"          # maps to daily_limit.max_native
    max_per_tx_native: "1.0"           # maps to spending_limit.max_native
```

:::note
Per-wallet policy management and richer policy configuration are not fully surfaced yet. Today Hermes supports:
- runtime RPC endpoint overrides via `wallet.rpc_endpoints`
- a minimal set of global agent-wallet policy overrides via `wallet.agent_wallet`
- durable freeze/rate-limit/daily-limit state across CLI invocations

More granular per-wallet policy editing is planned follow-up work.
:::
