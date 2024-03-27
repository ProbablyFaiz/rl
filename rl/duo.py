"""Adapted from:
- https://github.com/ivanov/duo-hotp/
- https://github.com/falsidge/ruo
"""
import base64
import dataclasses
import datetime
import email.utils
import enum
import urllib.parse
from typing import TypedDict

import requests
from Crypto.Hash import SHA512
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15


# Normally, I would make this a (str, Enum) subclass, but it turns out Python 3.11
#  makes a breaking change to (str, Enum) mixins! https://github.com/python/cpython/issues/100458
class DuoPath:
    ACTIVATION = "/push/v2/activation/{code}"
    TRANSACTIONS = "/push/v2/device/transactions"
    REPLY_TRANSACTION = "/push/v2/device/transactions/{transaction_id}"


APPROVAL_ANSWER = "approve"


class DuoConfig(TypedDict):
    host: str
    akey: str
    pkey: str
    hotp_secret: str
    hotp_counter: int
    key_pair: dict[str, str]


@dataclasses.dataclass
class Transaction:
    id: str


class DuoError(Exception):
    pass


@dataclasses.dataclass
class Duo:
    host: str
    akey: str
    pkey: str
    hotp_secret: str
    hotp_counter: int
    key: RSA

    @classmethod
    def from_qr_url(cls, qr_url: str) -> "Duo":
        qr_data = _parse_qr_url(qr_url)
        config = _activate_device(qr_data["host"], qr_data["code"])
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: DuoConfig) -> "Duo":
        return cls(
            host=config["host"],
            akey=config["akey"],
            pkey=config["pkey"],
            hotp_secret=config["hotp_secret"],
            hotp_counter=config["hotp_counter"],
            key=RSA.import_key(config["key_pair"]["private"]),
        )

    def to_config(self) -> DuoConfig:
        return {
            "host": self.host,
            "akey": self.akey,
            "pkey": self.pkey,
            "hotp_secret": self.hotp_secret,
            "hotp_counter": self.hotp_counter,
            "key_pair": {
                "public": self.key.public_key().export_key().decode(),
                "private": self.key.export_key().decode(),
            },
        }

    def answer_latest_transaction(self, approve: bool) -> dict:
        transactions = self.get_transactions()
        if not transactions:
            raise DuoError(f"No transactions found.")
        return self.answer_transaction(transactions[0].id, approve)

    def get_transactions(self) -> list[Transaction]:
        payload = {
            "akey": self.akey,
            "fips_status": "1",
            "hsm_status": "true",
            "pkpush": "rsa-sha512",
        }
        headers = self._get_headers(payload, DuoPath.TRANSACTIONS, "GET")
        url = _get_url(self.host, DuoPath.TRANSACTIONS)
        response = requests.get(url, params=payload, headers=headers)
        if response.status_code != 200:
            raise DuoError(f"Failed to get transactions: {response.text}")
        return [
            Transaction(id=c["urgid"])
            for c in response.json()["response"]["transactions"]
        ]

    def answer_transaction(self, transaction_id: str, approve: bool) -> dict:
        if approve:
            answer = APPROVAL_ANSWER
        else:
            raise NotImplementedError("Only approve is supported currently.")
        payload = {
            "akey": self.akey,
            "answer": answer,
            "fips_status": "1",
            "hsm_status": "true",
            "pkpush": "rsa-sha512",
        }
        path = DuoPath.REPLY_TRANSACTION.format(transaction_id=transaction_id)
        headers = self._get_headers(
            payload,
            path,
            "POST",
        )
        headers["txId"] = transaction_id  # I have no idea why this is needed
        url = _get_url(self.host, path)
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def _get_headers(self, data: dict, path: str, method: str) -> dict:
        time = email.utils.format_datetime(datetime.datetime.utcnow())
        message = (
            time
            + "\n"
            + method
            + "\n"
            + self.host.lower()
            + "\n"
            + path
            + "\n"
            + urllib.parse.urlencode(data)
        ).encode("ascii")
        signature = pkcs1_15.new(self.key).sign(SHA512.new(message))
        auth = "Basic " + b64_encode(f"{self.pkey}:{b64_encode(signature)}")
        return {
            "Authorization": auth,
            "x-duo-date": time,
            "host": self.host,
        }


def b64_encode(key: bytes | str) -> str:
    if isinstance(key, str):
        key = key.encode("ascii")
    return base64.b64encode(key).decode("ascii")


def _get_url(host: str, path: str) -> str:
    return f"https://{host}{path}"


def _parse_qr_url(qr_url: str) -> dict:
    value_param = urllib.parse.unquote(qr_url.split("?value=")[1])
    code, host = value_param.split("-")
    code = code.replace("duo://", "")
    host = base64.b64decode(host + "=" * (-len(host) % 4)).decode("utf-8")
    return {
        "code": code,
        "host": host,
    }


def _activate_device(host: str, code: str) -> DuoConfig:
    rsa_key = RSA.generate(2048)
    activation_url = _get_url(host, DuoPath.ACTIVATION.value.format(code=code))
    response = requests.post(
        activation_url,
        {
            "customer_protocol": "1",
            "pkpush": "rsa-sha512",
            "pubkey": (rsa_key.public_key().export_key("PEM").decode()),
            "jailbroken": "false",
            "Architecture": "arm64",
            "Legion": "US",
            "App_id": "com.duosecurity.duomobile",
            "full_disk_encryption": "true",
            "passcode_status": "true",
            "platform": "Android",
            "app_version": "3.49.0",
            "app_build_number": "323001",
            "version": "11",
            "manufacturer": "unknown",
            "language": "en",
            "model": "RL-CLI",
            "security_patch_level": "2021-02-01",
        },
    )
    response_dict = response.json()
    if response_dict["stat"] == "FAIL":
        raise Exception(
            f"Activation failed! Try a new QR/Activation URL. Response: {response_dict}"
        )

    return {
        "host": host,
        "akey": response_dict["response"]["akey"],
        "pkey": response_dict["response"]["pkey"],
        "hotp_secret": response_dict["response"]["hotp_secret"],
        "hotp_counter": 0,
        "key_pair": {
            "public": rsa_key.public_key().export_key().decode(),
            "private": rsa_key.export_key().decode(),
        },
    }
