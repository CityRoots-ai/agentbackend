#!/usr/bin/env python3
"""
Standalone Contract Verification Script for Etherscan
Usage: python verify_contract.py
"""

import json
import os
import requests
import time
from dotenv import load_dotenv

def verify_existing_contract():
    """Verify already deployed contract on Etherscan"""
    print("🔍 Contract Verification Tool")
    print("=" * 40)

    # Load environment variables
    load_dotenv()

    contract_address = os.getenv('CONTRACT_ADDRESS')
    etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')

    if not contract_address:
        print("❌ CONTRACT_ADDRESS not found in .env file")
        return

    if not etherscan_api_key:
        print("❌ ETHERSCAN_API_KEY not found in .env file")
        print("💡 Get a free API key from https://etherscan.io/apis and add to .env:")
        print("   ETHERSCAN_API_KEY=your_api_key_here")
        return

    print(f"📋 Contract Address: {contract_address}")
    print(f"🔑 Using API Key: {etherscan_api_key[:8]}...{etherscan_api_key[-4:]}")

    try:
        # Read the contract source code
        with open('contracts/CommunityVoting.sol', 'r') as f:
            source_code = f.read()

        # Prepare verification data
        verification_data = {
            'apikey': etherscan_api_key,
            'module': 'contract',
            'action': 'verifysourcecode',
            'contractaddress': contract_address,
            'sourceCode': source_code,
            'codeformat': 'solidity-single-file',
            'contractname': 'CommunityVoting',
            'compilerversion': 'v0.8.19+commit.7dd6d404',
            'optimizationUsed': '0',
            'runs': '200',
            'constructorArguements': '',
            'evmversion': '',
            'licenseType': '3'  # MIT License
        }

        # Submit verification
        print("\n📤 Submitting verification request to Sepolia Etherscan...")
        response = requests.post('https://api-sepolia.etherscan.io/api', data=verification_data)
        result = response.json()

        print(f"📊 API Response: {result}")

        if result['status'] == '1':
            guid = result['result']
            print(f"✅ Verification submitted successfully!")
            print(f"📋 Verification GUID: {guid}")

            # Check verification status
            print("\n⏳ Monitoring verification status...")
            for attempt in range(30):  # Wait up to 5 minutes
                time.sleep(10)
                print(f"🔄 Checking status... (attempt {attempt + 1}/30)")

                status_response = requests.get('https://api-sepolia.etherscan.io/api', params={
                    'apikey': etherscan_api_key,
                    'module': 'contract',
                    'action': 'checkverifystatus',
                    'guid': guid
                })

                status_result = status_response.json()
                print(f"📊 Status Response: {status_result}")

                if status_result['status'] == '1':
                    if status_result['result'] == 'Pass - Verified':
                        print("\n🎉 CONTRACT VERIFIED SUCCESSFULLY!")
                        print(f"🔗 View verified contract: https://sepolia.etherscan.io/address/{contract_address}#code")
                        print(f"📖 Source code is now publicly viewable")
                        print(f"⚡ Users can interact directly through Etherscan")
                        return
                    elif status_result['result'] == 'Fail - Unable to verify':
                        print(f"\n❌ VERIFICATION FAILED")
                        print(f"💡 Possible reasons:")
                        print(f"   - Compiler version mismatch")
                        print(f"   - Source code doesn't match deployed bytecode")
                        print(f"   - Constructor arguments missing/incorrect")
                        return
                    elif 'Pending' in status_result['result']:
                        print(f"⏳ Status: {status_result['result']}")
                    else:
                        print(f"🔄 Status: {status_result['result']}")
                else:
                    print(f"⚠️  API Error: {status_result}")

            print("\n⚠️  Verification timeout - please check manually on Etherscan")
            print(f"🔗 Manual check: https://sepolia.etherscan.io/address/{contract_address}")

        else:
            print(f"\n❌ VERIFICATION SUBMISSION FAILED")
            print(f"📊 Error: {result.get('result', 'Unknown error')}")
            if 'already verified' in str(result.get('result', '')).lower():
                print(f"✅ Contract might already be verified!")
                print(f"🔗 Check: https://sepolia.etherscan.io/address/{contract_address}#code")

    except FileNotFoundError:
        print("❌ Contract source file not found: contracts/CommunityVoting.sol")
    except Exception as e:
        print(f"❌ Verification error: {e}")

def check_verification_status():
    """Check if contract is already verified"""
    load_dotenv()

    contract_address = os.getenv('CONTRACT_ADDRESS')
    etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')

    if not contract_address or not etherscan_api_key:
        print("❌ Missing CONTRACT_ADDRESS or ETHERSCAN_API_KEY in .env")
        return

    try:
        # Check if contract source is available
        response = requests.get('https://api-sepolia.etherscan.io/api', params={
            'module': 'contract',
            'action': 'getsourcecode',
            'address': contract_address,
            'apikey': etherscan_api_key
        })

        result = response.json()

        if result['status'] == '1' and result['result'][0]['SourceCode']:
            print("✅ Contract is already verified!")
            print(f"🔗 View: https://sepolia.etherscan.io/address/{contract_address}#code")
            return True
        else:
            print("⚠️  Contract not yet verified")
            return False

    except Exception as e:
        print(f"❌ Error checking verification status: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking current verification status...")

    if check_verification_status():
        choice = input("\n📝 Contract already verified. Re-verify anyway? (y/n): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("✅ Verification check complete!")
            exit(0)

    verify_existing_contract()